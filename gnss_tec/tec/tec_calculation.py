from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Literal, overload

import numpy as np
import polars as pl
import polars.selectors as cs

from gnss_tec import read_rinex_obs

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
    lf: pl.LazyFrame, min_snr: float, bias_lf: pl.LazyFrame | None = None
) -> pl.LazyFrame:
    # ---- 1. Unpivot the LazyFrame to long format for easier processing. ----
    long_lf = (
        lf.select(pl.col("Time", "Station", "PRN"), cs.matches(r"^C\d[A-Z]$"))
        .unpivot(
            index=["Time", "Station", "PRN"], variable_name="Code", value_name="Value"
        )
        .drop_nulls("Value")
        .drop("Value")
    )

    # ---- 2. Map observation codes to bands (C1, C2). ----
    code_band: dict[str, str] = {}
    c1_priority: dict[str, int] = {}
    c2_priority: dict[str, int] = {}
    for const, codes in C1_CODES.items():
        for i, code in enumerate(codes):
            code_band[f"{const}_{code}"] = "C1"
            c1_priority[f"{const}_{code}"] = i
    for const, codes in C2_CODES.items():
        for i, code in enumerate(codes):
            code_band[f"{const}_{code}"] = "C2"
            c2_priority[f"{const}_{code}"] = i

    long_lf = (
        long_lf.with_columns(
            pl.col("PRN").cat.slice(0, 1).cast(pl.Categorical).alias("Constellation")
        )
        .with_columns(
            pl.concat_str(pl.col("Constellation"), pl.lit("_"), pl.col("Code"))
            .replace_strict(code_band, default=None)
            .alias("Band")
        )
        .drop_nulls("Band")
    )

    # ---- 3. Pivot back to wide format with resolved C bands as columns. ----
    resolved_lf = (
        long_lf.group_by("Time", "Station", "Constellation", "PRN")
        .agg(
            pl.col("Code").filter(pl.col("Band") == "C1").alias("C1_Code"),
            pl.col("Code").filter(pl.col("Band") == "C2").alias("C2_Code"),
        )
        .explode("C1_Code")
        .explode("C2_Code")
    )

    # ---- 4. Join bias data (if provided). ----
    if bias_lf is not None:
        bias_prn = bias_lf.filter(pl.col("STATION").is_null())
        bias_station = bias_lf.drop_nulls("STATION")
        resolved_lf = resolved_lf.join(
            bias_prn.select(
                "PRN", C1_Code="OBS1", C2_Code="OBS2", Bias_PRN="ESTIMATED_VALUE"
            ),
            on=["PRN", "C1_Code", "C2_Code"],
        ).join(
            bias_station.select(
                Station="STATION",
                Constellation="PRN",
                C1_Code="OBS1",
                C2_Code="OBS2",
                Bias_Station="ESTIMATED_VALUE",
            ),
            on=["Station", "Constellation", "C1_Code", "C2_Code"],
        )

    # ---- 5. Keep only the highest priority codes for C1 and C2. ----
    resolved_lf = resolved_lf.with_columns(
        pl.concat_str(pl.col("Constellation"), pl.lit("_"), pl.col("C1_Code"))
        .replace_strict(c1_priority, default=None)
        .alias("C1_Priority"),
        pl.concat_str(pl.col("Constellation"), pl.lit("_"), pl.col("C2_Code"))
        .replace_strict(c2_priority, default=None)
        .alias("C2_Priority"),
    ).select(
        pl.col(
            "Time", "Station", "PRN", "C1_Code", "C2_Code", "Bias_PRN", "Bias_Station"
        )
        .sort_by(pl.col("C1_Priority") * 2 + pl.col("C2_Priority"), descending=False)
        .first()
        .over("Time", "Station", "PRN", mapping_strategy="explode")
    )

    # ---- 6. Join back the observation values for the resolved codes. ----
    lf = lf.select(
        "Time", "Station", "PRN", "Azimuth", "Elevation", cs.matches(r"^[CLS]\d[A-Z]$")
    )

    def build_extract_expr(target_col_name: str, code_col_name: str) -> pl.Expr:
        """
        生成一个表达式：根据 code_col_name 列中存储的列名，去提取对应列的值。
        例如：如果 C1_Code 列的值是 "C1C"，则提取 "C1C" 列的值。
        """
        expr = None
        # 遍历所有可能的列名，构建 when-then 链
        available_cols = filter(
            lambda x: re.match(r"^[CLS]\d[A-Z]$", x), lf.collect_schema().names()
        )
        for col in available_cols:
            # 只有当 DataFrame 中实际存在该列时才处理
            cond = pl.col(code_col_name) == col
            val = pl.col(col)
            if expr is None:
                expr = pl.when(cond).then(val)
            else:
                expr = expr.when(cond).then(val)
        if expr is None:
            return pl.lit(None).alias(target_col_name)
        return expr.otherwise(None).alias(target_col_name)

    resolved_lf = (
        resolved_lf.join(lf, on=["Time", "Station", "PRN"])
        .with_columns(
            pl.col("C1_Code").str.slice(1, None).alias("C1_Band"),
            pl.col("C2_Code").str.slice(1, None).alias("C2_Band"),
        )
        .with_columns(
            pl.concat_str(pl.lit("L"), pl.col("C1_Band")).alias("L1_Code"),
            pl.concat_str(pl.lit("L"), pl.col("C2_Band")).alias("L2_Code"),
            pl.concat_str(pl.lit("S"), pl.col("C1_Band")).alias("S1_Code"),
            pl.concat_str(pl.lit("S"), pl.col("C2_Band")).alias("S2_Code"),
        )
        .drop("C1_Band", "C2_Band")
        .with_columns(
            build_extract_expr("C1", "C1_Code"),
            build_extract_expr("C2", "C2_Code"),
            build_extract_expr("L1", "L1_Code"),
            build_extract_expr("L2", "L2_Code"),
            build_extract_expr("S1", "S1_Code"),
            build_extract_expr("S2", "S2_Code"),
        )
        .drop(cs.matches(r"^[A-Z]\d[A-Z]$"), cs.matches(r"^[LS][12]_Code$"))
        .filter(pl.col("S1") >= min_snr, pl.col("S2") >= min_snr)
        .drop("S1", "S2")
    )

    return resolved_lf


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

    return lf.with_columns(
        pl.concat_str(
            pl.col("PRN").cat.slice(0, 1),
            pl.lit("_"),
            pl.col("C1_Code").str.slice(1, 1),
        )
        .replace_strict(code_freq_map, default=None)
        .alias("C1_Freq"),
        pl.concat_str(
            pl.col("PRN").cat.slice(0, 1),
            pl.lit("_"),
            pl.col("C2_Code").str.slice(1, 1),
        )
        .replace_strict(code_freq_map, default=None)
        .alias("C2_Freq"),
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
    rx_bias: Literal["external", "msd", "fallback-msd"] = "fallback-msd",
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
    rx_bias: Literal["external", "msd", "fallback-msd"] = "fallback-msd",
    constellations: str | None = None,
    t_lim: tuple[str | None, str | None] | list[str | None] | None = None,
    *,
    lazy: Literal[False] = False,
) -> pl.DataFrame: ...


def calc_tec(
    obs_fn: str | Path | Iterable[str | Path],
    nav_fn: str | Path | Iterable[str | Path],
    bias_fn: str | Path | Iterable[str | Path] | None = None,
    rx_bias: Literal["external", "msd", "fallback-msd"] = "fallback-msd",
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
        rx_bias (Literal["external", "msd", "fallback-msd"], optional): Method for
            receiver bias correction. Possible values are,
            - "external": Use biases only from the provided bias file(s).
            - "msd": Use the Minimum Standard Deviation (MSD) method to estimate.
            - "fallback-msd": Use biases from the provided bias file(s) if available.
              Otherwise, apply the MSD method.
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
    lf = lf.filter(pl.col("Elevation") >= min_elevation).drop(cs.matches(r"^D\d[A-Z]$"))

    if bias_fn is not None:
        from .bias import read_bias

        bias_lf = read_bias(bias_fn, lazy=True).with_columns(
            pl.col("PRN").cast(pl.Categorical), pl.col("STATION").cast(pl.Categorical)
        )
    else:
        bias_lf = None

    lf = lf.with_columns(
        pl.col("Station").cast(pl.Categorical), pl.col("PRN").cast(pl.Categorical)
    )
    lf = _resolve_observations_and_bias(lf, min_snr, bias_lf)
    lf = _map_frequencies(lf)

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

    rx_lat, rx_lon, _ = header.rx_geodetic
    mf, ipp_lat, ipp_lon = _single_layer_model(
        pl.col("Azimuth"), pl.col("Elevation"), pl.lit(rx_lat), pl.lit(rx_lon)
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
        .with_columns(pl.col("Station").cast(pl.String), pl.col("PRN").cast(pl.String))
    )

    if lazy:
        return lf
    else:
        return lf.collect()
