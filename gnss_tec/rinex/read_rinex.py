from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, overload

import pandas as pd
import polars as pl
import pymap3d as pm

from .._core import _read_obs

ALL_CONSTELLATIONS = {
    "C": "BDS",
    "G": "GPS",
    "E": "Galileo",
    "R": "GLONASS",
    "J": "QZSS",
    "I": "IRNSS",
    "S": "SBAS",
}
"""All supported GNSS constellations for RINEX file reading."""


@dataclass
class RinexObsHeader:
    """Dataclass for RINEX observation file header metadata."""

    version: str
    """RINEX version."""

    constellation: str | None
    """Constellation for which the RINEX file contains observations."""

    marker_name: str
    """Marker name."""

    marker_type: str | None
    """Marker type."""

    rx_ecef: tuple[float, float, float]
    """Approximate receiver position in ECEF coordinates (X, Y, Z) in meters."""

    rx_geodetic: tuple[float, float, float]
    """Approximate receiver position in geodetic coordinates (latitude, longitude,
        altitude) in degrees and meters."""

    sampling_interval: int | None
    """Sampling interval in seconds."""

    leap_seconds: int | None
    """Number of leap seconds."""


def _handle_fn(fn: str | Path | Iterable[str | Path]) -> list[str]:
    if isinstance(fn, (str, Path)):
        fn_list = [str(fn)]
    elif isinstance(fn, Iterable):
        fn_list = [str(f) for f in fn]
    else:
        raise TypeError(
            f"The file path must be a str, Path, or Iterable of str/Path, not {fn}."
        )

    for f in fn_list:
        if not Path(f).exists():
            raise FileNotFoundError(f"RINEX file not found: {f}")

    return fn_list


def _handle_t_str(t_str: str | None) -> str | None:
    if t_str is None:
        return None

    iso_format = "%Y-%m-%d %H:%M:%S"

    if t_str.strip().upper().endswith("GPST"):
        return pd.to_datetime(t_str.replace("GPST", "")).strftime(iso_format) + " GPST"
    else:
        dt = pd.to_datetime(t_str)
        if dt.tzinfo is None:
            return dt.strftime(iso_format) + " UTC"
        else:
            return dt.tz_convert("UTC").strftime(iso_format) + " UTC"


@overload
def read_rinex_obs(
    obs_fn: str | Path | Iterable[str | Path],
    nav_fn: str | Path | Iterable[str | Path] | None = None,
    constellations: str | None = None,
    t_lim: tuple[str | None, str | None] | list[str | None] | None = None,
    codes: Iterable[str] | None = None,
    *,
    lazy: Literal[True],
) -> tuple[RinexObsHeader, pl.LazyFrame]: ...


@overload
def read_rinex_obs(
    obs_fn: str | Path | Iterable[str | Path],
    nav_fn: str | Path | Iterable[str | Path] | None = None,
    constellations: str | None = None,
    t_lim: tuple[str | None, str | None] | list[str | None] | None = None,
    codes: Iterable[str] | None = None,
    *,
    lazy: Literal[False] = False,
) -> tuple[RinexObsHeader, pl.DataFrame]: ...


def read_rinex_obs(
    obs_fn: str | Path | Iterable[str | Path],
    nav_fn: str | Path | Iterable[str | Path] | None = None,
    constellations: str | None = None,
    t_lim: tuple[str | None, str | None] | list[str | None] | None = None,
    codes: Iterable[str] | None = None,
    *,
    lazy: bool = False,
) -> tuple[RinexObsHeader, pl.DataFrame | pl.LazyFrame]:
    """Read RINEX observation file into a Polars DataFrame.

    Args:
        obs_fn (str | Path | Iterable[str | Path]): Path(s) to the RINEX observation
            file(s). These files must be from the same station, otherwise the output
            DataFrame will be incorrect.
        nav_fn (str | Path | Iterable[str | Path] | None, optional): Path(s) to the
            RINEX navigation file(s). If provided, azimuth and elevation angles will be
            computed. Defaults to None.
        constellations (str | None, optional): String of constellation codes to filter
            by. If None, all supported constellations are included. See
            `gnss_tec.rinex.ALL_CONSTELLATIONS` for valid codes. Defaults to None.
        t_lim (tuple[str | None, str | None] | list[str | None] | None, optional): Time
            limits for filtering observations. Should be a tuple or list with two
            elements representing the start and end times. Use None for no limit on
            either end. Timezone can be specified using ISO 8601 format (e.g.,
            '2023-01-01 00:00:00Z', '2023-01-01 00:00:00+0800', as long as
            `pd.to_datetime` can parse it). Additionally, 'GPST' is also supported
            (e.g., '2023-01-01 00:00:00 GPST'). If no timezone is provided, UTC is
            assumed. Defaults to None.
        codes (Iterable[str] | None, optional): Specific observation codes to extract
            (e.g., ['C1C', 'L1C']). If None, all available observation types are
            included. Defaults to None.
        lazy (bool, optional): Whether to return a `polars.LazyFrame`. Defaults to
            False.

    Returns:
        (RinexObsHeader, pl.DataFrame | pl.LazyFrame): A Dataclass containing metadata from the RINEX observation file header and a DataFrame or LazyFrame containing the RINEX observation data with following columns.
            - Time (datetime): Observation timestamp.
            - Station (str): 4-character station identifier.
            - PRN (str): Satellite PRN identifier.
            - Azimuth (float): Satellite azimuth angle in degrees (if nav_fn provided).
            - Elevation (float): Satellite elevation angle in degrees (if nav_fn
                provided).
            - Observations (float): Various observation types (e.g., C1C, C1X,
                L1C, S1C), each as a separate column.

    Raises:
        FileNotFoundError: If the observation or navigation file does not exist.
        ValueError: If an unknown constellation code is provided.
    """
    obs_fn_list = _handle_fn(obs_fn)
    if nav_fn is not None:
        nav_fn_list = _handle_fn(nav_fn)
    else:
        nav_fn_list = None

    if constellations is not None:
        constellations = constellations.upper()
        for c in constellations:
            if c not in ALL_CONSTELLATIONS:
                raise ValueError(
                    f"Unknown constellation code: {c}. "
                    f"Valid codes are: {', '.join(ALL_CONSTELLATIONS.keys())}"
                )

    if t_lim is None:
        t_lim = [None, None]
    else:
        t_lim = [_handle_t_str(t_lim[0]), _handle_t_str(t_lim[1])]

    result: dict = _read_obs(
        obs_fn_list,
        nav_fn=nav_fn_list,
        constellations=constellations,
        t_lim=tuple(t_lim),
        codes=codes if codes is None else list(set(codes)),
    )
    codes = list(filter(lambda x: re.match(r"[A-Z]\d{1}[A-Z]$", x), result.keys()))
    ordered_cols = ["Time", "Station", "PRN"]
    rx_x = result.pop("RX_X")
    rx_y = result.pop("RX_Y")
    rx_z = result.pop("RX_Z")
    rx_lat, rx_lon, rx_alt = pm.ecef2geodetic(rx_x, rx_y, rx_z, deg=True)
    if nav_fn is not None:
        ordered_cols += ["Azimuth", "Elevation"]
        nav_x = result.pop("NAV_X")
        nav_y = result.pop("NAV_Y")
        nav_z = result.pop("NAV_Z")
        result["Azimuth"], result["Elevation"], _ = pm.ecef2aer(
            nav_x, nav_y, nav_z, rx_lat, rx_lon, rx_alt, deg=True
        )
    ordered_cols += sorted(codes)

    header = RinexObsHeader(
        version=result.pop("Version"),
        constellation=result.pop("Constellation"),
        marker_name=result["Station"],
        marker_type=result.pop("MarkerType"),
        rx_ecef=(rx_x, rx_y, rx_z),
        rx_geodetic=(float(rx_lat), float(rx_lon), float(rx_alt)),
        sampling_interval=result.pop("SamplingInterval"),
        leap_seconds=result.pop("LeapSeconds"),
    )

    df = (
        pl.DataFrame(result)
        .lazy()
        .with_columns(pl.col("Time").cast(pl.Datetime("ms", "UTC")))
        .fill_nan(None)
        .sort(["Time", "Station", "PRN"])
        .select(ordered_cols)
    )

    if lazy:
        return header, df
    else:
        return header, df.collect()
