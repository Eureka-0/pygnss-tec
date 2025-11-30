from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import pymap3d as pm

from .._core import _get_nav_coords, _read_obs

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


def read_rinex_obs(
    obs_fn: str | Path, constellations: str | None = None, include_doppler: bool = True
) -> tuple[RinexObsHeader, pl.LazyFrame]:
    """Read RINEX observation file into a Polars DataFrame.

    Args:
        obs_fn (str | Path): Path to the RINEX observation file.
        constellations (str | None, optional): String of constellation codes to filter
            by. If None, all supported constellations are included. See
            `gnss_tec.rinex.ALL_CONSTELLATIONS` for valid codes. Defaults to None.
        include_doppler (bool, optional): Whether to include Doppler observations.
            Defaults to True.
        lazy (bool, optional): Whether to return a `polars.LazyFrame`. Defaults to
            False.

    Returns:
        (RinexObsHeader, pl.LazyFrame): A Dataclass containing metadata from the RINEX
        observation file header and a LazyFrame containing the RINEX observation data.

    Raises:
        ValueError: If an unknown constellation code is provided.
    """
    if constellations is not None:
        constellations = constellations.upper()
        for c in constellations:
            if c not in ALL_CONSTELLATIONS:
                raise ValueError(
                    f"Unknown constellation code: {c}. "
                    f"Valid codes are: {', '.join(ALL_CONSTELLATIONS.keys())}"
                )

    header_dict, time, prn, code, value = _read_obs(
        str(obs_fn), constellations=constellations, include_doppler=include_doppler
    )

    rx_x = header_dict["rx_x"]
    rx_y = header_dict["rx_y"]
    rx_z = header_dict["rx_z"]
    rx_lat, rx_lon, rx_alt = pm.ecef2geodetic(rx_x, rx_y, rx_z, deg=True)
    header = RinexObsHeader(
        version=header_dict["version"],
        constellation=header_dict["constellation"],
        marker_name=header_dict["station"],
        marker_type=header_dict["marker_type"],
        rx_ecef=(rx_x, rx_y, rx_z),
        rx_geodetic=(float(rx_lat), float(rx_lon), float(rx_alt)),
        sampling_interval=header_dict["sampling_interval"],
        leap_seconds=header_dict["leap_seconds"],
    )

    lf = (
        pl.DataFrame(
            {
                "time": pl.Series(time),
                "prn": pl.Series(prn),
                "code": pl.Series(code),
                "value": pl.Series(value),
            }
        )
        .lazy()
        .with_columns(pl.col("time").cast(pl.Datetime("ms", "UTC")))
        .fill_nan(None)
    )
    return header, lf


def get_nav_coords(
    nav_fn: str | Path | Iterable[str | Path], df: pl.DataFrame | pl.LazyFrame
) -> pl.LazyFrame:
    """
    Get satellite ECEF coordinates from RINEX navigation file(s).

    Args:
        nav_fn (str | Path | Iterable[str | Path]): Path(s) to the RINEX navigation
            file(s).
        df (pl.DataFrame | pl.LazyFrame): DataFrame or LazyFrame containing 'time' and
            'prn' columns.

    Returns:
        pl.LazyFrame: LazyFrame with columns 'nav_x', 'nav_y', 'nav_z' representing
            satellite ECEF coordinates in meters.
    """
    nav_fn_list = _handle_fn(nav_fn)

    def map_nav_coords(df: pl.DataFrame) -> pl.DataFrame:
        time = df["time"].dt.epoch("ms").cast(pl.Int64).to_arrow()
        prn = df["prn"].to_arrow()
        nav_x, nav_y, nav_z = _get_nav_coords(nav_fn=nav_fn_list, time=time, prn=prn)
        df_coords = pl.DataFrame(
            {
                "nav_x": pl.Series(nav_x),
                "nav_y": pl.Series(nav_y),
                "nav_z": pl.Series(nav_z),
            }
        )
        return pl.concat([df, df_coords], how="horizontal")

    schema = df.collect_schema()
    schema.update({"nav_x": pl.Float64(), "nav_y": pl.Float64(), "nav_z": pl.Float64()})
    return df.lazy().map_batches(map_nav_coords, schema=schema)
