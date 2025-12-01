from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Literal

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
        lf.select(pl.col("time", "station", "prn"), cs.matches(r"^C\d[A-Z]$"))
        .unpivot(
            index=["time", "station", "prn"], variable_name="code", value_name="value"
        )
        .drop_nulls("value")
        .drop("value")
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
            pl.col("prn").cat.slice(0, 1).cast(pl.Categorical).alias("constellation")
        )
        .with_columns(
            pl.concat_str(pl.col("constellation"), pl.lit("_"), pl.col("code"))
            .replace_strict(code_band, default=None)
            .alias("band")
        )
        .drop_nulls("band")
    )

    # ---- 3. Pivot back to wide format with resolved C bands as columns. ----
    resolved_lf = (
        long_lf.group_by("time", "station", "constellation", "prn")
        .agg(
            pl.col("code").filter(pl.col("band") == "C1").alias("C1_code"),
            pl.col("code").filter(pl.col("band") == "C2").alias("C2_code"),
        )
        .explode("C1_code")
        .explode("C2_code")
    )

    # ---- 4. Join bias data (if provided). ----
    if bias_lf is not None:
        bias_prn = bias_lf.filter(pl.col("station").is_null())
        bias_station = bias_lf.drop_nulls("station")
        resolved_lf = resolved_lf.join(
            bias_prn.select(
                "prn", C1_code="obs1", C2_code="obs2", bias_prn="estimated_value"
            ),
            on=["prn", "C1_code", "C2_code"],
        ).join(
            bias_station.select(
                "station",
                constellation="prn",
                C1_code="obs1",
                C2_code="obs2",
                bias_station="estimated_value",
            ),
            on=["station", "constellation", "C1_code", "C2_code"],
        )

    # ---- 5. Keep only the highest priority codes for C1 and C2. ----
    resolved_lf = resolved_lf.with_columns(
        pl.concat_str(pl.col("constellation"), pl.lit("_"), pl.col("C1_code"))
        .replace_strict(c1_priority, default=None)
        .alias("C1_priority"),
        pl.concat_str(pl.col("constellation"), pl.lit("_"), pl.col("C2_code"))
        .replace_strict(c2_priority, default=None)
        .alias("C2_priority"),
    ).select(
        pl.col(
            "time", "station", "prn", "C1_code", "C2_code", "bias_prn", "bias_station"
        )
        .sort_by(pl.col("C1_priority") * 2 + pl.col("C2_priority"), descending=False)
        .first()
        .over("time", "station", "prn", mapping_strategy="explode")
    )

    # ---- 6. Join back the observation values for the resolved codes. ----
    lf = lf.select(
        "time", "station", "prn", "azimuth", "elevation", cs.matches(r"^[CLS]\d[A-Z]$")
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
        resolved_lf.join(lf, on=["time", "station", "prn"])
        .with_columns(
            pl.col("C1_code").str.slice(1, None).alias("C1_band"),
            pl.col("C2_code").str.slice(1, None).alias("C2_band"),
        )
        .with_columns(
            pl.concat_str(pl.lit("L"), pl.col("C1_band")).alias("L1_code"),
            pl.concat_str(pl.lit("L"), pl.col("C2_band")).alias("L2_code"),
            pl.concat_str(pl.lit("S"), pl.col("C1_band")).alias("S1_code"),
            pl.concat_str(pl.lit("S"), pl.col("C2_band")).alias("S2_code"),
        )
        .drop("C1_band", "C2_band")
        .with_columns(
            build_extract_expr("C1", "C1_code"),
            build_extract_expr("C2", "C2_code"),
            build_extract_expr("L1", "L1_code"),
            build_extract_expr("L2", "L2_code"),
            build_extract_expr("S1", "S1_code"),
            build_extract_expr("S2", "S2_code"),
        )
        .drop(cs.matches(r"^[A-Z]\d[A-Z]$"), cs.matches(r"^[LS][12]_code$"))
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
            pl.col("prn").cat.slice(0, 1),
            pl.lit("_"),
            pl.col("C1_code").str.slice(1, 1),
        )
        .replace_strict(code_freq_map, default=None)
        .alias("C1_freq"),
        pl.concat_str(
            pl.col("prn").cat.slice(0, 1),
            pl.lit("_"),
            pl.col("C2_code").str.slice(1, 1),
        )
        .replace_strict(code_freq_map, default=None)
        .alias("C2_freq"),
    ).drop("C1_code", "C2_code")


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


def calc_tec(
    obs_fn: str | Path | Iterable[str | Path],
    nav_fn: str | Path | Iterable[str | Path],
    bias_fn: str | Path | Iterable[str | Path] | None = None,
    rx_bias: Literal["external", "msd", "fallback-msd"] = "fallback-msd",
    constellations: str | None = None,
    min_elevation: float = DEFAULT_MIN_ELEVATION,
    min_snr: float = DEFAULT_MIN_SNR,
) -> pl.LazyFrame:
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
        min_elevation (float, optional): Minimum satellite elevation angle in degrees
            for including observations. Defaults to DEFAULT_MIN_ELEVATION (40.0).
        min_snr (float, optional): Minimum signal-to-noise ratio in dB-Hz for
            including observations. Defaults to DEFAULT_MIN_SNR (30.0).
        lazy (bool, optional): Whether to return a `polars.LazyFrame`. Defaults to
            False.

    Returns:
        pl.LazyFrame: A LazyFrame containing the calculated TEC values.
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

    header, lf = read_rinex_obs(obs_fn, nav_fn, constellations)
    if header.sampling_interval == 1:
        arc_interval = pl.duration(minutes=1)
    else:
        arc_interval = pl.duration(minutes=5)

    # Filter by minimum elevation angle and drop D-code observations
    lf = lf.filter(pl.col("elevation") >= min_elevation).drop(cs.matches(r"^D\d[A-Z]$"))

    if bias_fn is not None:
        from .bias import read_bias

        bias_lf = read_bias(bias_fn, lazy=True).with_columns(
            pl.col("prn").cast(pl.Categorical), pl.col("station").cast(pl.Categorical)
        )
    else:
        bias_lf = None

    lf = lf.with_columns(
        pl.col("station").cast(pl.Categorical), pl.col("prn").cast(pl.Categorical)
    )
    lf = _resolve_observations_and_bias(lf, min_snr, bias_lf)
    lf = _map_frequencies(lf)

    f1 = pl.col("C1_freq")
    f2 = pl.col("C2_freq")
    coeff = f1**2 * f2**2 / (f1**2 - f2**2) / 40.3e16
    tecu_per_ns = 1e-9 * c * coeff

    lf = (
        lf.with_columns(
            # sTEC from pseudorange, in TECU
            (pl.col("C2") - pl.col("C1")).mul(coeff).alias("stec_g"),
            # sTEC from carrier phase, in TECU
            (pl.col("L1") / f1 - pl.col("L2") / f2).mul(c * coeff).alias("stec_p"),
        )
        .with_columns(
            (pl.col("stec_g") - pl.col("stec_p")).alias("raw_offset"),
            pl.col("elevation").radians().sin().pow(2).alias("weight"),
            pl.col("time")
            .diff()
            .ge(arc_interval)
            .fill_null(False)
            .cum_sum()
            .over("station", "prn")
            .alias("arc_id"),
        )
        .with_columns(
            pl.col("raw_offset")
            .mul(pl.col("weight"))
            .sum()
            .truediv(pl.col("weight").sum())
            .over("station", "prn", "arc_id")
            .alias("offset")
        )
        .with_columns(
            # levelled sTEC, in TECU
            (pl.col("stec_p") + pl.col("offset")).alias("stec")
        )
        .drop("C1", "C2", "L1", "L2", "raw_offset", "weight", "offset", "arc_id")
    )

    rx_lat, rx_lon, _ = header.rx_geodetic
    mf, ipp_lat, ipp_lon = _single_layer_model(
        pl.col("azimuth"), pl.col("elevation"), pl.lit(rx_lat), pl.lit(rx_lon)
    )

    if bias_fn is not None:
        lf = (
            lf.with_columns(
                # total DCB biases, in TECU
                (pl.col("bias_prn") + pl.col("bias_station"))
                .mul(tecu_per_ns)
                .alias("bias")
            )
            .with_columns(
                # sTEC corrected for DCB biases, in TECU
                (pl.col("stec") + pl.col("bias")).alias("stec_bias_corrected")
            )
            .with_columns(
                # vTEC, in TECU
                (pl.col("stec_bias_corrected") / mf).alias("vtec")
            )
            .drop("bias_prn", "bias_station")
        )
    else:
        lf = lf.with_columns(
            # vTEC, in TECU
            (pl.col("stec") / mf).alias("vtec")
        )

    lf = (
        lf.with_columns(
            # IPP Latitude in degrees
            ipp_lat.alias("ipp_lat"),
            # IPP Longitude in degrees
            ipp_lon.alias("ipp_lon"),
        )
        .drop("C1_freq", "C2_freq", "azimuth")
        .sort("time", "station", "prn")
        .with_columns(pl.col("station").cast(pl.String), pl.col("prn").cast(pl.String))
    )

    return lf
