from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

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

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "constellation": self.constellation,
            "marker_name": self.marker_name,
            "marker_type": self.marker_type,
            "rx_ecef": self.rx_ecef,
            "rx_geodetic": self.rx_geodetic,
            "sampling_interval": self.sampling_interval,
            "leap_seconds": self.leap_seconds,
        }

    def to_metadata(self) -> dict[str, str]:
        """
        Convert the RinexObsHeader to a dictionary suitable for storing as Parquet file
        metadata using Polars.
        """
        return {
            "version": self.version,
            "constellation": self.constellation or "",
            "marker_name": self.marker_name,
            "marker_type": self.marker_type or "",
            "rx_ecef_x": str(self.rx_ecef[0]),
            "rx_ecef_y": str(self.rx_ecef[1]),
            "rx_ecef_z": str(self.rx_ecef[2]),
            "rx_geodetic_lat": str(self.rx_geodetic[0]),
            "rx_geodetic_lon": str(self.rx_geodetic[1]),
            "rx_geodetic_alt": str(self.rx_geodetic[2]),
            "sampling_interval": str(self.sampling_interval or ""),
            "leap_seconds": str(self.leap_seconds or ""),
        }


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
    obs_fn: str | Path | Iterable[str | Path],
    nav_fn: str | Path | Iterable[str | Path] | None = None,
    constellations: str | None = None,
    codes: Iterable[str] | None = None,
    *,
    utc: bool = True,
    station: str | None = None,
    pivot: bool = True,
) -> tuple[RinexObsHeader, pl.LazyFrame]:
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
        codes (Iterable[str] | None, optional): Specific observation codes to extract
            (e.g., ['C1C', 'L1C']). If None, all available observation types are
            included. Defaults to None.
        utc (bool, optional): Whether to convert time to UTC. If False, time will be in
            GPS time. In this case, ensure that leap seconds **are consistent** across
            all input files. Defaults to True.
        station (str | None, optional): Station name to assign to the DataFrame. If
            None, the station name from the RINEX header is used. Defaults to None.
        pivot (bool, optional): Whether to pivot the DataFrame so that each observation
            type has its own column. If False, the DataFrame will be in long format with
            'code' and 'value' columns. Pivoted format is generally more convenient for
            analysis and has better performance. Defaults to True.

    Returns:
        (RinexObsHeader, pl.LazyFrame): A Dataclass containing metadata from the RINEX
            observation file header and a LazyFrame containing the RINEX observation
            data.

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

    header_dict, batch = _read_obs(
        obs_fn_list,
        nav_fn=nav_fn_list,
        constellations=constellations,
        codes=None if codes is None else list(set(codes)),
        pivot=pivot,
    )
    codes = list(filter(lambda x: re.match(r"[A-Z]\d{1}[A-Z]$", x), batch.schema.names))
    ordered_cols = ["time", "station", "prn"]
    rx_x = header_dict["rx_x"]
    rx_y = header_dict["rx_y"]
    rx_z = header_dict["rx_z"]
    rx_lat, rx_lon, rx_alt = pm.ecef2geodetic(rx_x, rx_y, rx_z, deg=True)

    header = RinexObsHeader(
        version=header_dict["version"],
        constellation=header_dict["constellation"],
        marker_name=header_dict["station"][:4].strip() if station is None else station,
        marker_type=header_dict["marker_type"],
        rx_ecef=(rx_x, rx_y, rx_z),
        rx_geodetic=(float(rx_lat), float(rx_lon), float(rx_alt)),
        sampling_interval=header_dict["sampling_interval"],
        leap_seconds=header_dict["leap_seconds"],
    )

    df = pl.DataFrame(batch)

    def calc_az_el(df: pl.DataFrame) -> pl.DataFrame:
        az, el, _ = pm.ecef2aer(
            df.get_column("sat_x"),
            df.get_column("sat_y"),
            df.get_column("sat_z"),
            rx_lat,
            rx_lon,
            rx_alt,
            deg=True,
        )
        return df.with_columns(
            pl.Series("azimuth", az, dtype=pl.Float32),
            pl.Series("elevation", el, dtype=pl.Float32),
        )

    if nav_fn is not None:
        ordered_cols += ["azimuth", "elevation"]
        df = df.pipe(calc_az_el)
    if pivot:
        ordered_cols += sorted(codes)
    else:
        ordered_cols += ["code", "value"]

    lf = (
        df.lazy()
        .with_columns(
            pl.col("time").cast(pl.Datetime("ms", "UTC")),
            pl.lit(header.marker_name).cast(pl.Categorical).alias("station"),
            pl.col("prn").cast(pl.Categorical),
        )
        .fill_nan(None)
        .select(ordered_cols)
        .sort(["time", "station", "prn"])
    )

    if not utc:
        lf = lf.with_columns(
            pl.col("time")
            .add(
                get_leap_seconds("time")
                if header.leap_seconds is None
                else pl.duration(seconds=header.leap_seconds)
            )
            .dt.replace_time_zone(None)
        )

    return header, lf


def get_leap_seconds(time_col: pl.Expr | str) -> pl.Expr:
    if isinstance(time_col, str):
        time_col = pl.col(time_col)

    return (
        pl.when(
            time_col.is_between(
                pl.datetime(1980, 1, 1), pl.datetime(1981, 7, 1), "left"
            )
        )
        .then(pl.duration(seconds=0))
        .when(
            time_col.is_between(
                pl.datetime(1981, 7, 1), pl.datetime(1982, 7, 1), "left"
            )
        )
        .then(pl.duration(seconds=1))
        .when(
            time_col.is_between(
                pl.datetime(1982, 7, 1), pl.datetime(1983, 7, 1), "left"
            )
        )
        .then(pl.duration(seconds=2))
        .when(
            time_col.is_between(
                pl.datetime(1983, 7, 1), pl.datetime(1985, 7, 1), "left"
            )
        )
        .then(pl.duration(seconds=3))
        .when(
            time_col.is_between(
                pl.datetime(1985, 7, 1), pl.datetime(1988, 1, 1), "left"
            )
        )
        .then(pl.duration(seconds=4))
        .when(
            time_col.is_between(
                pl.datetime(1988, 1, 1), pl.datetime(1990, 1, 1), "left"
            )
        )
        .then(pl.duration(seconds=5))
        .when(
            time_col.is_between(
                pl.datetime(1990, 1, 1), pl.datetime(1991, 1, 1), "left"
            )
        )
        .then(pl.duration(seconds=6))
        .when(
            time_col.is_between(
                pl.datetime(1991, 1, 1), pl.datetime(1992, 7, 1), "left"
            )
        )
        .then(pl.duration(seconds=7))
        .when(
            time_col.is_between(
                pl.datetime(1992, 7, 1), pl.datetime(1993, 7, 1), "left"
            )
        )
        .then(pl.duration(seconds=8))
        .when(
            time_col.is_between(
                pl.datetime(1993, 7, 1), pl.datetime(1994, 7, 1), "left"
            )
        )
        .then(pl.duration(seconds=9))
        .when(
            time_col.is_between(
                pl.datetime(1994, 7, 1), pl.datetime(1996, 1, 1), "left"
            )
        )
        .then(pl.duration(seconds=10))
        .when(
            time_col.is_between(
                pl.datetime(1996, 1, 1), pl.datetime(1997, 7, 1), "left"
            )
        )
        .then(pl.duration(seconds=11))
        .when(
            time_col.is_between(
                pl.datetime(1997, 7, 1), pl.datetime(1999, 1, 1), "left"
            )
        )
        .then(pl.duration(seconds=12))
        .when(
            time_col.is_between(
                pl.datetime(1999, 1, 1), pl.datetime(2006, 1, 1), "left"
            )
        )
        .then(pl.duration(seconds=13))
        .when(
            time_col.is_between(
                pl.datetime(2006, 1, 1), pl.datetime(2009, 1, 1), "left"
            )
        )
        .then(pl.duration(seconds=14))
        .when(
            time_col.is_between(
                pl.datetime(2009, 1, 1), pl.datetime(2012, 7, 1), "left"
            )
        )
        .then(pl.duration(seconds=15))
        .when(
            time_col.is_between(
                pl.datetime(2012, 7, 1), pl.datetime(2015, 7, 1), "left"
            )
        )
        .then(pl.duration(seconds=16))
        .when(
            time_col.is_between(
                pl.datetime(2015, 7, 1), pl.datetime(2016, 12, 31), "left"
            )
        )
        .then(pl.duration(seconds=17))
        .when(time_col.ge(pl.datetime(2016, 12, 31)))
        .then(pl.duration(seconds=18))
        .otherwise(None)
    )
