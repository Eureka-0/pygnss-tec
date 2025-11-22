from __future__ import annotations

import re
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


def _handle_fn(fn: str | Path | Iterable[str | Path]) -> list[str]:
    if isinstance(fn, (str, Path)):
        fn_list = [str(fn)]
    elif isinstance(fn, Iterable):
        fn_list = [str(f) for f in fn]
    else:
        raise TypeError(
            f"The file path must be a str, Path, or an iterable of str/Path, not {fn}."
        )
    for f in fn_list:
        if not Path(f).exists():
            raise FileNotFoundError(f"RINEX file not found: {f}")
    return fn_list


@overload
def read_rinex_obs(
    obs_fn: str | Path | Iterable[str | Path],
    lazy: Literal[True],
    nav_fn: str | Path | Iterable[str | Path] | None = None,
    constellations: str | None = None,
    t_lim: tuple[str | None, str | None] | list[str | None] | None = None,
    codes: Iterable[str] | None = None,
) -> pl.LazyFrame: ...


@overload
def read_rinex_obs(
    obs_fn: str | Path | Iterable[str | Path],
    lazy: Literal[False] = False,
    nav_fn: str | Path | Iterable[str | Path] | None = None,
    constellations: str | None = None,
    t_lim: tuple[str | None, str | None] | list[str | None] | None = None,
    codes: Iterable[str] | None = None,
) -> pl.DataFrame: ...


def read_rinex_obs(
    obs_fn: str | Path | Iterable[str | Path],
    lazy: bool = False,
    nav_fn: str | Path | Iterable[str | Path] | None = None,
    constellations: str | None = None,
    t_lim: tuple[str | None, str | None] | list[str | None] | None = None,
    codes: Iterable[str] | None = None,
) -> pl.DataFrame | pl.LazyFrame:
    """Read RINEX observation file into a Polars DataFrame.

    Args:
        obs_fn (str | Path | Iterable[str | Path]): Path(s) to the RINEX observation
            file(s). These files must be from the same station, otherwise the output
            DataFrame will be incorrect.
        lazy (bool, optional): If True, returns a Polars LazyFrame for deferred
            computation. If False, returns a Polars DataFrame. Defaults to False.
        nav_fn (str | Path | Iterable[str | Path] | None, optional): Path(s) to the
            RINEX navigation file(s). If provided, azimuth and elevation angles will be
            computed. Defaults to None.
        constellations (str | None, optional): String of constellation codes to filter
            by. Valid codes are: 'C' for BDS, 'G' for GPS, 'E' for Galileo, 'R' for
            GLONASS, 'J' for QZSS, 'I' for IRNSS, 'S' for SBAS. If None, all
            constellations are included. Defaults to None.
        t_lim (tuple[str | None, str | None] | list[str | None] | None, optional): Time
            limits for filtering observations. Should be a tuple or list with two
            elements representing the start and end times. Use None for no limit on
            either end. Defaults to None.
        codes (Iterable[str] | None, optional): Specific observation codes to extract
            (e.g., ['C1C', 'L1C']). If None, all available observation types are
            included. Defaults to None.

    Returns:
        pl.DataFrame | pl.LazyFrame: DataFrame or LazyFrame containing the RINEX
            observation data with following columns.
            - Time (datetime): Observation timestamp.
            - Station (str): 4-character station identifier.
            - PRN (str): Satellite PRN identifier.
            - RX_LAT (float): Receiver latitude in degrees.
            - RX_LON (float): Receiver longitude in degrees.
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
        t_lim = [t_lim[0], t_lim[1]]
    for i, t in enumerate(t_lim):
        if isinstance(t, str):
            tz = re.search(r"[A-Z]{3,4}$", t)
            form = "%Y-%m-%dT%H:%M:%S"
            if tz is None:
                t_str = pd.to_datetime(t).strftime(form)
                t_lim[i] = t_str + " UTC"
            else:
                t_str = pd.to_datetime(t[: -len(tz.group(0))]).strftime(form)
                t_lim[i] = t_str + " " + tz.group(0)

    result: dict = _read_obs(
        obs_fn_list,
        nav_fn=nav_fn_list,
        constellations=constellations,
        t_lim=tuple(t_lim),
        codes=codes if codes is None else list(set(codes)),
    )
    codes = list(filter(lambda x: re.match(r"[A-Z]\d{1}[A-Z]$", x), result.keys()))
    ordered_cols = ["Time", "Station", "PRN", "RX_LAT", "RX_LON"]
    rx_x = result.pop("RX_X")
    rx_y = result.pop("RX_Y")
    rx_z = result.pop("RX_Z")
    result["RX_LAT"], result["RX_LON"], rx_alt = pm.ecef2geodetic(
        rx_x, rx_y, rx_z, deg=True
    )
    if nav_fn is not None:
        ordered_cols += ["Azimuth", "Elevation"]
        nav_x = result.pop("NAV_X")
        nav_y = result.pop("NAV_Y")
        nav_z = result.pop("NAV_Z")
        result["Azimuth"], result["Elevation"], _ = pm.ecef2aer(
            nav_x, nav_y, nav_z, result["RX_LAT"], result["RX_LON"], rx_alt, deg=True
        )
    ordered_cols += sorted(codes)

    df = (
        pl.DataFrame(result)
        .lazy()
        .with_columns(pl.col("Time").cast(pl.Datetime("ms", "UTC")))
        .fill_nan(None)
        .sort(["Time", "Station", "PRN"])
        .select(ordered_cols)
    )

    if lazy:
        return df
    else:
        return df.collect()
