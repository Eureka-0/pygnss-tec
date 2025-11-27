from __future__ import annotations

from pathlib import Path
from typing import Iterable, Literal, overload

import numpy as np
import polars as pl
import polars.selectors as cs

from ..rinex import read_rinex_obs
from .constants import (
    C1_CODES,
    C2_CODES,
    DEFAULT_IPP_HEIGHT,
    SIGNAL_FREQ,
    SUPPORTED_CONSTELLATIONS,
    Re,
    c,
)


def _resolve_observations(lf: pl.LazyFrame) -> pl.LazyFrame:
    def get_valid_codes(c_codes: dict[str, list[str]]):
        columns = lf.collect_schema().names()
        valid_c_codes = {
            constellation: list(filter(lambda c: c in columns, code_list))
            for constellation, code_list in c_codes.items()
        }
        valid_l_codes = {
            constellation: [f"L{c[1:]}" for c in codes]
            for constellation, codes in valid_c_codes.items()
        }
        return valid_c_codes, valid_l_codes

    def resolve_value(codes_dict: dict[str, list[str]]) -> pl.Expr:
        return pl.coalesce(
            pl.when(pl.col("PRN").str.starts_with(constellation)).then(pl.col(codes))
            for constellation, codes in codes_dict.items()
        )

    def resolve_code(codes_dict: dict[str, list[str]]) -> pl.Expr:
        return pl.coalesce(
            [
                pl.when(pl.col("PRN").str.starts_with(constellation)).then(
                    pl.when(pl.col(code).is_not_null()).then(pl.lit(code))
                )
                for constellation, codes in codes_dict.items()
                for code in codes
            ]
        )

    valid_c1_codes, valid_l1_codes = get_valid_codes(C1_CODES)
    valid_c2_codes, valid_l2_codes = get_valid_codes(C2_CODES)
    return (
        lf.with_columns(
            resolve_value(valid_c1_codes).alias("C1"),
            resolve_value(valid_c2_codes).alias("C2"),
            resolve_code(valid_c1_codes).alias("C1_Code"),
            resolve_code(valid_c2_codes).alias("C2_Code"),
            resolve_value(valid_l1_codes).alias("L1"),
            resolve_value(valid_l2_codes).alias("L2"),
            resolve_code(valid_l1_codes).alias("L1_Code"),
            resolve_code(valid_l2_codes).alias("L2_Code"),
        )
        .drop(cs.matches(r"^[A-Z]\d{1}[A-Z]$"))
        .drop_nulls(["C1", "C2", "L1", "L2"])
    )


def _map_frequencies(lf: pl.LazyFrame) -> pl.LazyFrame:
    code_freq_map = {
        "C_1": SIGNAL_FREQ["C"]["B1"],
        "C_2": SIGNAL_FREQ["C"]["B1-2"],
        "C_5": SIGNAL_FREQ["C"]["B2a"],
        "C_6": SIGNAL_FREQ["C"]["B3"],
        "C_7": SIGNAL_FREQ["C"]["B2b"],
        "C_8": SIGNAL_FREQ["C"]["B2"],
        "G_1": SIGNAL_FREQ["G"]["L1"],
        "G_2": SIGNAL_FREQ["G"]["L2"],
        "G_5": SIGNAL_FREQ["G"]["L5"],
    }

    def freq_col(code_col: str) -> pl.Expr:
        expr = pl.when(False).then(None)
        for constellation in SUPPORTED_CONSTELLATIONS:
            expr = expr.when(pl.col("PRN").str.starts_with(constellation)).then(
                pl.col(code_col)
                .str.slice(1, 1)
                .str.pad_start(2, "_")
                .str.pad_start(3, constellation)
                .replace_strict(code_freq_map, default=None)
            )
        return expr

    return lf.with_columns(
        freq_col("C1_Code").alias("C1_Freq"),
        freq_col("C2_Code").alias("C2_Freq"),
        freq_col("L1_Code").alias("L1_Freq"),
        freq_col("L2_Code").alias("L2_Freq"),
    ).drop("C1_Code", "C2_Code", "L1_Code", "L2_Code")


def _single_layer_model(
    azimuth: pl.Expr, elevation: pl.Expr, rx_lat_deg: pl.Expr, rx_lon_deg: pl.Expr
) -> tuple[pl.Expr, pl.Expr, pl.Expr]:
    """
    Calculate the mapping function and Ionospheric Pierce Point (IPP) latitude and
        longitude using the Single Layer Model (SLM).

    Args:
        azimuth (pl.Expr): Satellite azimuth angle in degrees.
        elevation (pl.Expr): Satellite elevation angle in degrees.
        rx_lat_deg (pl.Expr): Receiver latitude in degrees.
        rx_lon_deg (pl.Expr): Receiver longitude in degrees.

    Returns:
        tuple[pl.Expr, pl.Expr, pl.Expr]: A tuple containing:
            - Mapping function (pl.Expr)
            - IPP latitude in degrees (pl.Expr)
            - IPP longitude in degrees (pl.Expr)
    """
    az = azimuth.radians()
    el = elevation.radians()
    rx_lat = rx_lat_deg.radians()
    rx_lon = rx_lon_deg.radians()

    # mapping function
    sin_beta = Re * el.cos() / (Re + DEFAULT_IPP_HEIGHT)
    mf = sin_beta.arcsin().cos().pow(-1)

    # IPP latitude and longitude, in radians
    psi = np.pi / 2 - el - sin_beta.arcsin()
    ipp_lat = (rx_lat.sin() * psi.cos() + rx_lat.cos() * psi.sin() * az.cos()).arcsin()
    ipp_lon = rx_lon + (psi.sin() * az.sin() / ipp_lat.cos()).arcsin()

    return mf, ipp_lat.degrees(), ipp_lon.degrees()


@overload
def calc_tec(
    obs_fn: str | Path | Iterable[str | Path],
    nav_fn: str | Path | Iterable[str | Path],
    constellations: str | None = None,
    t_lim: tuple[str | None, str | None] | list[str | None] | None = None,
    *,
    lazy: Literal[True],
) -> pl.LazyFrame: ...


@overload
def calc_tec(
    obs_fn: str | Path | Iterable[str | Path],
    nav_fn: str | Path | Iterable[str | Path],
    constellations: str | None = None,
    t_lim: tuple[str | None, str | None] | list[str | None] | None = None,
    *,
    lazy: Literal[False] = False,
) -> pl.DataFrame: ...


def calc_tec(
    obs_fn: str | Path | Iterable[str | Path],
    nav_fn: str | Path | Iterable[str | Path],
    constellations: str | None = None,
    t_lim: tuple[str | None, str | None] | list[str | None] | None = None,
    *,
    lazy: bool = False,
) -> pl.LazyFrame | pl.DataFrame:
    """
    Calculate the Total Electron Content (TEC) from RINEX observation and navigation files.

    Args:
        obs_fn (str | Path | Iterable[str | Path]): Path(s) to the RINEX observation
            file(s). These files must be from the same station, otherwise the output
            DataFrame will be incorrect.
        nav_fn (str | Path | Iterable[str | Path]): Path(s) to the RINEX navigation
            file(s).
        constellations (str | None, optional): Constellations to consider. If None, all
            supported constellations are used. Defaults to None.
        t_lim (tuple[str | None, str | None] | list[str | None] | None, optional): Time
            limits for TEC calculation. Should be a tuple or list with two
            elements representing the start and end times. Use None for no limit on
            either end. Timezone can be specified using ISO 8601 format (e.g.,
            '2023-01-01 00:00:00Z', '2023-01-01 00:00:00+0800', as long as
            `pd.to_datetime` can parse it). Additionally, 'GPST' is also supported
            (e.g., '2023-01-01 00:00:00 GPST'). If no timezone is provided, UTC is
            assumed. Defaults to None.
        lazy (bool, optional): Whether to return a `polars.LazyFrame`. Defaults to
            False.

    Returns:
        (pl.LazyFrame | pl.DataFrame): A LazyFrame or DataFrame containing the
            calculated TEC values.
    """
    if constellations is not None:
        constellations = constellations.upper()
        for con in constellations:
            if con not in SUPPORTED_CONSTELLATIONS:
                raise NotImplementedError(
                    f"Constellation '{con}' is not supported for TEC calculation. "
                    f"Supported constellations are: {SUPPORTED_CONSTELLATIONS}"
                )

    header, lf = read_rinex_obs(obs_fn, nav_fn, constellations, t_lim, lazy=True)
    rx_lat_deg, rx_lon_deg, _ = header.rx_geodetic
    mf, ipp_lat, ipp_lon = _single_layer_model(
        pl.col("Azimuth"), pl.col("Elevation"), pl.lit(rx_lat_deg), pl.lit(rx_lon_deg)
    )

    lf = _resolve_observations(lf)
    lf = _map_frequencies(lf)

    def multiplier(f1: pl.Expr, f2: pl.Expr) -> pl.Expr:
        return f1**2 * f2**2 / (f1**2 - f2**2) / 40.3e16

    lf = (
        lf.with_columns(
            # sTEC from pseudorange, in TECU
            (pl.col("C2") - pl.col("C1"))
            .mul(multiplier(pl.col("C1_Freq"), pl.col("C2_Freq")))
            .alias("sTEC_g"),
            # sTEC from carrier phase, in TECU
            (pl.col("L1") / pl.col("L1_Freq") - pl.col("L2") / pl.col("L2_Freq"))
            .mul(c * multiplier(pl.col("L1_Freq"), pl.col("L2_Freq")))
            .alias("sTEC_p"),
        )
        .drop("C1", "C2", "L1", "L2", "C1_Freq", "C2_Freq", "L1_Freq", "L2_Freq")
        .with_columns(
            (pl.col("sTEC_g") - pl.col("sTEC_p")).alias("Raw_Offset"),
            pl.col("Elevation").radians().sin().pow(2).alias("Weight"),
        )
        .with_columns(
            pl.col("Raw_Offset")
            .mul(pl.col("Weight"))
            .sum()
            .truediv(pl.col("Weight").sum())
            .over("PRN")
            .alias("Offset")
        )
        .drop("Raw_Offset", "Weight")
        .with_columns(
            # levelled sTEC, in TECU
            (pl.col("sTEC_p") + pl.col("Offset")).alias("sTEC")
        )
        .drop("sTEC_g", "sTEC_p", "Offset")
        .with_columns(
            # vTEC, in TECU
            (pl.col("sTEC") / mf).alias("vTEC"),
            # IPP Latitude in degrees
            ipp_lat.alias("IPP_Lat"),
            # IPP Longitude in degrees
            ipp_lon.alias("IPP_Lon"),
        )
    )

    if lazy:
        return lf
    else:
        return lf.collect()
