# PyGNSS-TEC

[![PyPI - Version](https://img.shields.io/pypi/v/pygnss-tec)](https://pypi.org/project/pygnss-tec/)
![Supported Python Versions](https://img.shields.io/badge/python-%3E%3D3.9-blue)
![License](https://img.shields.io/badge/license-MIT-blue)
[![Pytest](https://github.com/Eureka-0/pygnss-tec/actions/workflows/pytest.yml/badge.svg)](https://github.com/Eureka-0/pygnss-tec/actions/workflows/pytest.yml)

PyGNSS-TEC is a high-performance Python package leveraging Rust acceleration, designed for processing and analyzing Total Electron Content (TEC) data derived from Global Navigation Satellite System (GNSS) observations. The package provides tools for RINEX file reading, TEC calculation, and DCB correction to support ionospheric studies.

> **Warning**: This package is under active development and may undergo significant changes. It is not recommended for production use until it reaches a stable release (v1.0.0).

## Features

- **RINEX File Reading**: Efficient reading and parsing of RINEX GNSS observation files using [rinex crate](https://crates.io/crates/rinex) (see [benchmarks](#benchmarks) for details).

- **Multiple File Formats**: Support for RINEX versions 2.x and 3.x., as well as Hatanaka compressed files (e.g., .Z, .crx, .crx.gz).

- **TEC Calculation**: Efficiently compute TEC from dual-frequency GNSS observations using [polars](https://pola.rs/) DataFrames and lazy evaluation (see [benchmarks](#benchmarks) for details).

- **Multi-GNSS Support**: Process observations from multiple GNSS constellations - GPS and Beidou (with full RINEX reading and TEC calculation support), and Galileo, GLONASS, QZSS, IRNSS, and SBAS (RINEX reading only).

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
```

The built package will be available in the `target/wheels` directory. You can then install it to your Python environment or uv project with:

```bash
# Using pip
pip install target/wheels/pygnss_tec-*.whl

# Or using uv
uv pip install target/wheels/pygnss_tec-*.whl
```

## Usage

## Benchmarks
