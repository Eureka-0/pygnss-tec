from __future__ import annotations

from pathlib import Path
from typing import Iterable, Literal, overload

import numpy as np
import polars as pl

from ..rinex import read_rinex_obs
from .constants import (
    C1_CODES,
    C2_CODES,
    DEFAULT_IPP_HEIGHT,
    DEFAULT_MIN_ELEVATION,
    DEFAULT_MIN_SNR,
    SIGNAL_FREQ,
    SUPPORTED_CONSTELLATIONS,
    Re,
    c,
)


def _resolve_observations_and_bias(
    lf: pl.LazyFrame, bias_lf: pl.LazyFrame | None = None
) -> pl.LazyFrame:
    # ---- 1. Get valid observation codes (C) present in the LazyFrame. ----
    columns = lf.collect_schema().names()

    def get_valid_codes(c_codes: dict[str, list[str]]) -> dict[str, list[str]]:
        valid_codes = {
            constellation: list(filter(lambda c: c in columns, code_list))
            for constellation, code_list in c_codes.items()
        }
        return valid_codes

    valid_c1_codes = get_valid_codes(C1_CODES)
    valid_c2_codes = get_valid_codes(C2_CODES)

    # ---- 2. Unpivot the LazyFrame to long format for easier processing. ----
    lf = (
        lf.unpivot(
            index=["Time", "Station", "PRN", "Azimuth", "Elevation"],
            variable_name="Code",
            value_name="Value",
        )
        .drop_nulls("Value")
        .with_columns(pl.col("PRN").str.slice(0, 1).alias("Constellation"))
    )

    # ---- 3. Map observation codes to bands (C1, C2, L1, L2, S1, S2). ----
    code2band: dict[str, str] = {}
    c12priority: dict[str, int] = {}
    c22priority: dict[str, int] = {}
    for const, codes in valid_c1_codes.items():
        for i, code in enumerate(codes):
            code2band[f"{const}_{code}"] = "C1"
            c12priority[f"{const}_{code}"] = i
            code2band[f"{const}_L{code[1:]}"] = "L1"
            code2band[f"{const}_S{code[1:]}"] = "S1"
    for const, codes in valid_c2_codes.items():
        for i, code in enumerate(codes):
            code2band[f"{const}_{code}"] = "C2"
            c22priority[f"{const}_{code}"] = i
            code2band[f"{const}_L{code[1:]}"] = "L2"
            code2band[f"{const}_S{code[1:]}"] = "S2"

    lf = lf.with_columns(
        pl.concat_str(pl.col("Constellation"), pl.lit("_"), pl.col("Code"))
        .replace_strict(code2band, default=None)
        .alias("Band")
    ).drop_nulls("Band")

    # ---- 4. Pivot back to wide format with resolved bands as columns. ----
    lf = (
        lf.group_by("Time", "Station", "PRN", "Azimuth", "Elevation")
        .agg(
            pl.col("Code").filter(pl.col("Band") == "C1").alias("C1_Code"),
            pl.col("Value").filter(pl.col("Band") == "C1").alias("C1"),
            pl.col("Code").filter(pl.col("Band") == "C2").alias("C2_Code"),
            pl.col("Value").filter(pl.col("Band") == "C2").alias("C2"),
            pl.col("Code").filter(pl.col("Band") == "L1").alias("L1_Code"),
            pl.col("Value").filter(pl.col("Band") == "L1").alias("L1"),
            pl.col("Code").filter(pl.col("Band") == "L2").alias("L2_Code"),
            pl.col("Value").filter(pl.col("Band") == "L2").alias("L2"),
            pl.col("Code").filter(pl.col("Band") == "S1").alias("S1_Code"),
            pl.col("Value").filter(pl.col("Band") == "S1").alias("S1"),
            pl.col("Code").filter(pl.col("Band") == "S2").alias("S2_Code"),
            pl.col("Value").filter(pl.col("Band") == "S2").alias("S2"),
        )
        .explode("C1_Code", "C1")
        .explode("L1_Code", "L1")
        .filter(pl.col("C1_Code").str.slice(1, 2) == pl.col("L1_Code").str.slice(1, 2))
        .explode("C2_Code", "C2")
        .explode("L2_Code", "L2")
        .filter(pl.col("C2_Code").str.slice(1, 2) == pl.col("L2_Code").str.slice(1, 2))
        .explode("S1_Code", "S1")
        .explode("S2_Code", "S2")
        .filter(
            pl.col("C1_Code").str.slice(1, 2) == pl.col("S1_Code").str.slice(1, 2),
            pl.col("C2_Code").str.slice(1, 2) == pl.col("S2_Code").str.slice(1, 2),
        )
        .drop("L1_Code", "L2_Code", "S1_Code", "S2_Code")
        .with_columns(pl.col("PRN").str.slice(0, 1).alias("Constellation"))
        .with_columns(
            pl.concat_str(pl.col("Constellation"), pl.lit("_"), pl.col("C1_Code"))
            .replace_strict(c12priority, default=None)
            .alias("C1_Priority"),
            pl.concat_str(pl.col("Constellation"), pl.lit("_"), pl.col("C2_Code"))
            .replace_strict(c22priority, default=None)
            .alias("C2_Priority"),
        )
    )

    # ---- 5. Only keep the codes that are in the bias file (if provided). ----
    if bias_lf is not None:
        bias_prn = bias_lf.filter(pl.col("STATION").is_null())
        bias_station = bias_lf.drop_nulls("STATION")
        lf = (
            lf.join(bias_prn.select("PRN", "OBS1", "OBS2", "ESTIMATED_VALUE"), on="PRN")
            .filter(
                pl.col("C1_Code") == pl.col("OBS1"), pl.col("C2_Code") == pl.col("OBS2")
            )
            .rename({"ESTIMATED_VALUE": "Bias_PRN"})
            .drop("OBS1", "OBS2")
            .join(
                bias_station.select(
                    "STATION", "PRN", "OBS1", "OBS2", "ESTIMATED_VALUE"
                ).rename(
                    {
                        "STATION": "Station",
                        "PRN": "Constellation",
                        "ESTIMATED_VALUE": "Bias_Station",
                    }
                ),
                on=["Station", "Constellation"],
            )
            .filter(
                pl.col("C1_Code") == pl.col("OBS1"), pl.col("C2_Code") == pl.col("OBS2")
            )
            .drop("OBS1", "OBS2")
        )

    # ---- 6. Keep only the highest priority codes for C1 and C2. ----
    lf = (
        lf.group_by("Time", "Station", "PRN", "Azimuth", "Elevation")
        .agg(
            pl.all()
            .sort_by(
                pl.col("C1_Priority") * 2 + pl.col("C2_Priority"), descending=False
            )
            .first()
        )
        .drop("C1_Priority", "C2_Priority", "Constellation")
    )

    return lf


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
        freq_col("C1_Code").alias("C1_Freq"), freq_col("C2_Code").alias("C2_Freq")
    ).drop("C1_Code", "C2_Code")


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
    bias_fn: str | Path | Iterable[str | Path] | None = None,
    constellations: str | None = None,
    t_lim: tuple[str | None, str | None] | list[str | None] | None = None,
    *,
    lazy: Literal[True],
) -> pl.LazyFrame: ...


@overload
def calc_tec(
    obs_fn: str | Path | Iterable[str | Path],
    nav_fn: str | Path | Iterable[str | Path],
    bias_fn: str | Path | Iterable[str | Path] | None = None,
    constellations: str | None = None,
    t_lim: tuple[str | None, str | None] | list[str | None] | None = None,
    *,
    lazy: Literal[False] = False,
) -> pl.DataFrame: ...


def calc_tec(
    obs_fn: str | Path | Iterable[str | Path],
    nav_fn: str | Path | Iterable[str | Path],
    bias_fn: str | Path | Iterable[str | Path] | None = None,
    constellations: str | None = None,
    t_lim: tuple[str | None, str | None] | list[str | None] | None = None,
    min_elevation: float = DEFAULT_MIN_ELEVATION,
    min_snr: float = DEFAULT_MIN_SNR,
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
        bias_fn (str | Path | Iterable[str | Path] | None, optional): Path(s) to the
            bias file(s). If provided, DCB biases will be applied to the TEC
            calculation. Defaults to None.
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
        min_elevation (float, optional): Minimum satellite elevation angle in degrees
            for including observations. Defaults to DEFAULT_MIN_ELEVATION (40.0).
        min_snr (float, optional): Minimum signal-to-noise ratio in dB-Hz for
            including observations. Defaults to DEFAULT_MIN_SNR (30.0).
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
    else:
        constellations = "".join(SUPPORTED_CONSTELLATIONS.keys())

    header, lf = read_rinex_obs(obs_fn, nav_fn, constellations, t_lim, lazy=True)
    rx_lat_deg, rx_lon_deg, _ = header.rx_geodetic
    mf, ipp_lat, ipp_lon = _single_layer_model(
        pl.col("Azimuth"), pl.col("Elevation"), pl.lit(rx_lat_deg), pl.lit(rx_lon_deg)
    )

    if bias_fn is not None:
        from .bias import read_bias

        bias_lf = read_bias(bias_fn, lazy=True)
    else:
        bias_lf = None

    lf = _resolve_observations_and_bias(lf, bias_lf)
    lf = _map_frequencies(lf)
    lf = lf.filter(
        pl.col("Elevation") >= min_elevation,
        pl.col("S1") >= min_snr,
        pl.col("S2") >= min_snr,
    ).drop("S1", "S2")

    f1 = pl.col("C1_Freq")
    f2 = pl.col("C2_Freq")
    coeff = f1**2 * f2**2 / (f1**2 - f2**2) / 40.3e16
    tecu_per_ns = 1e-9 * c * coeff

    lf = (
        lf.with_columns(
            # sTEC from pseudorange, in TECU
            (pl.col("C2") - pl.col("C1")).mul(coeff).alias("sTEC_g"),
            # sTEC from carrier phase, in TECU
            (pl.col("L1") / f1 - pl.col("L2") / f2).mul(c * coeff).alias("sTEC_p"),
        )
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
        .with_columns(
            # levelled sTEC, in TECU
            (pl.col("sTEC_p") + pl.col("Offset")).alias("sTEC")
        )
        .drop("C1", "C2", "L1", "L2", "Raw_Offset", "Weight", "Offset")
    )

    if bias_fn is not None:
        lf = (
            lf.with_columns(
                # total DCB biases, in TECU
                (pl.col("Bias_PRN") + pl.col("Bias_Station"))
                .mul(tecu_per_ns)
                .alias("Bias")
            )
            .with_columns(
                # sTEC corrected for DCB biases, in TECU
                (pl.col("sTEC") + pl.col("Bias")).alias("sTEC_bias_corrected")
            )
            .with_columns(
                # vTEC, in TECU
                (pl.col("sTEC_bias_corrected") / mf).alias("vTEC")
            )
            .drop("Bias_PRN", "Bias_Station")
        )
    else:
        lf = lf.with_columns(
            # vTEC, in TECU
            (pl.col("sTEC") / mf).alias("vTEC")
        )

    lf = (
        lf.with_columns(
            # IPP Latitude in degrees
            ipp_lat.alias("IPP_Lat"),
            # IPP Longitude in degrees
            ipp_lon.alias("IPP_Lon"),
        )
        .drop("C1_Freq", "C2_Freq", "Azimuth")
        .sort("Time", "Station", "PRN")
    )

    if lazy:
        return lf
    else:
        return lf.collect()
