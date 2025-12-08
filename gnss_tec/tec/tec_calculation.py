from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import polars as pl
import polars.selectors as cs

from ..rinex import get_leap_seconds, read_rinex_obs
from .bias import read_bias
from .constants import SIGNAL_FREQ, TECConfig, c, get_sampling_config
from .mapping_func import single_layer_model


def _coalesce_observations(
    lf: pl.LazyFrame, bias_lf: pl.LazyFrame | None, rx_bias: bool, config: TECConfig
) -> pl.LazyFrame:
    # --- 1. Prepare coalesced observation codes for C1 and C2. ---
    lf = lf.with_columns(pl.col("time").dt.date().alias("date"))
    codes_lf = (
        lf.select("date", "station", "prn", cs.matches(r"^C\d[A-Z]$"))
        .group_by("date", "station", "prn")
        .agg(pl.all().null_count() / pl.len())
        .unpivot(
            index=["date", "station", "prn"], variable_name="code", value_name="null_pc"
        )
        .filter(pl.col("null_pc") <= 0.05)
        .drop("null_pc")
        .with_columns(
            pl.col("prn").cat.slice(0, 1).cast(pl.Categorical).alias("constellation"),
            pl.col("code").cast(pl.Categorical),
        )
        .sort(["date", "station", "prn"])
        .with_columns(
            pl.concat_str(pl.col("constellation"), pl.lit("_"), pl.col("code"))
            .replace_strict(config.code2band, default=None, return_dtype=pl.UInt8)
            .alias("band")
        )
        .drop_nulls("band")
        .group_by("date", "station", "constellation", "prn")
        .agg(
            pl.col("code").filter(pl.col("band") == 1).alias("C1_code"),
            pl.col("code").filter(pl.col("band") == 2).alias("C2_code"),
        )
        .explode("C1_code")
        .explode("C2_code")
    )

    # ---- 2. Join bias data (if provided). ----
    if bias_lf is not None:
        tx_bias_lf = bias_lf.filter(pl.col("station").is_null()).select(
            "prn",
            C1_code="obs1",
            C2_code="obs2",
            date=pl.col("bias_start").dt.date(),
            tx_bias=-pl.col("estimated_value"),
        )
        codes_lf = codes_lf.join(tx_bias_lf, on=["prn", "C1_code", "C2_code", "date"])

        if rx_bias:
            rx_bias_lf = bias_lf.drop_nulls("station").select(
                "station",
                constellation="prn",
                C1_code="obs1",
                C2_code="obs2",
                date=pl.col("bias_start").dt.date(),
                rx_bias=-pl.col("estimated_value"),
            )
            codes_lf = codes_lf.join(
                rx_bias_lf,
                on=["station", "constellation", "C1_code", "C2_code", "date"],
            )
        else:
            codes_lf = codes_lf.with_columns(pl.lit(None).alias("rx_bias"))

    # ---- 3. Keep only the highest priority codes for C1 and C2. ----
    codes_lf = (
        codes_lf.with_columns(
            pl.concat_str(pl.col("constellation"), pl.lit("_"), pl.col("C1_code"))
            .replace_strict(config.c1_priority, default=None, return_dtype=pl.UInt8)
            .alias("C1_priority"),
            pl.concat_str(pl.col("constellation"), pl.lit("_"), pl.col("C2_code"))
            .replace_strict(config.c2_priority, default=None, return_dtype=pl.UInt8)
            .alias("C2_priority"),
        )
        .group_by("date", "station", "prn")
        .agg(
            pl.all()
            .sort_by(
                pl.col("C1_priority") * 2 + pl.col("C2_priority"), descending=False
            )
            .first()
        )
    )

    # ---- 4. Join back the observation values for the coalesced codes. ----
    def build_extract_expr(code_col_name: str) -> pl.Expr:
        """
        Generate an expression that extracts the value from the column named in
        `code_col_name`.

        For example, if the value in the C1_Code column is "C1C", extract the value
        from the "C1C" column.
        """
        expr = None
        # Iterate over all possible column names to build a when-then chain
        available_cols = filter(
            lambda x: re.match(rf"^[{code_col_name[0]}]\d[A-Z]$", x),
            lf.collect_schema().names(),
        )
        for col in available_cols:
            cond = pl.col(code_col_name) == col
            val = pl.col(col)
            if expr is None:
                expr = pl.when(cond).then(val)
            else:
                expr = expr.when(cond).then(val)
        if expr is None:
            return pl.lit(None)
        return expr.otherwise(None)

    return (
        lf.join(
            codes_lf.drop("constellation", "C1_priority", "C2_priority"),
            on=["date", "station", "prn"],
            how="left",
        )
        .drop("date")
        .with_columns(
            pl.col("C1_code").cat.slice(1, None).cast(pl.Categorical).alias("C1_band"),
            pl.col("C2_code").cat.slice(1, None).cast(pl.Categorical).alias("C2_band"),
        )
        .with_columns(
            pl.concat_str(pl.lit("L"), pl.col("C1_band")).alias("L1_code"),
            pl.concat_str(pl.lit("L"), pl.col("C2_band")).alias("L2_code"),
            pl.concat_str(pl.lit("S"), pl.col("C1_band")).alias("S1_code"),
            pl.concat_str(pl.lit("S"), pl.col("C2_band")).alias("S2_code"),
        )
        .drop("C1_band", "C2_band")
        .with_columns(
            build_extract_expr("S1_code").alias("S1"),
            build_extract_expr("S2_code").alias("S2"),
        )
        .with_columns(
            (pl.col("S1").null_count() / pl.len()).cast(pl.Float32).alias("S1_null_pc"),
            (pl.col("S2").null_count() / pl.len()).cast(pl.Float32).alias("S2_null_pc"),
        )
        .filter(
            (pl.col("S1") >= config.min_snr) | (pl.col("S1_null_pc") > 0.5),
            (pl.col("S2") >= config.min_snr) | (pl.col("S2_null_pc") > 0.5),
        )
        .drop("S1", "S2", "S1_null_pc", "S2_null_pc")
        .with_columns(
            build_extract_expr("C1_code").alias("C1"),
            build_extract_expr("C2_code").alias("C2"),
            build_extract_expr("L1_code").alias("L1"),
            build_extract_expr("L2_code").alias("L2"),
        )
        .drop(cs.matches(r"^[A-Z]\d[A-Z]$"), cs.matches(r"^[LS][12]_code$"))
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

    def map_freq(col: str) -> pl.Expr:
        return pl.concat_str(
            pl.col("prn").cat.slice(0, 1), pl.lit("_"), pl.col(col).cat.slice(1, 1)
        ).replace_strict(code_freq_map, default=None)

    return lf.with_columns(
        map_freq("C1_code").alias("C1_freq"), map_freq("C2_code").alias("C2_freq")
    )


def _correct_cycle_slip(
    time: np.ndarray, stec_p: np.ndarray, tec_diff_tol: float, window_size: int
):
    """
    Correct the cycle slips in the sTEC from carrier phase.

    Args:
        - time (np.ndarray): the time array, in seconds.
        - stec_p (np.ndarray): the sTEC from carrier phase, in TECU.
        - tec_diff_tol (float): the threshold of the TEC difference, in TECU.
        - window_size (int): the size of the window for the interpolation.

    Returns:
        - np.ndarray: the cycle-slip-corrected sTEC, in TECU.
    """
    stec_p_corrected = stec_p.copy()
    start = -1
    end = -1

    for i, tec in enumerate(stec_p_corrected):
        if i < window_size:
            continue

        tec_diff = abs(tec - stec_p_corrected[i - 1])
        slipped = tec_diff > tec_diff_tol
        if start == -1 and slipped:
            start = i
        elif start != -1 and slipped:
            end = i

        if start != -1 and i == len(stec_p_corrected) - 1:
            end = len(stec_p_corrected)

        if start != -1 and end != -1:
            window = slice(start - window_size, start)
            tp = np.interp(time[start], time[window], stec_p_corrected[window])
            offset = tp - stec_p_corrected[start]
            stec_p_corrected[start:end] += offset

            tec_diff = abs(tec - stec_p_corrected[i - 1])
            slipped = tec_diff > tec_diff_tol
            start = i if slipped else -1
            end = -1

    return stec_p_corrected


def calc_tec_from_df(
    df: pl.DataFrame | pl.LazyFrame,
    sampling_interval: int | None = None,
    bias_fn: str | Path | Iterable[str | Path] | None = None,
    config: TECConfig = TECConfig(),
    *,
    rx_bias: bool = True,
) -> pl.LazyFrame:
    """
    Calculate the Total Electron Content (TEC) from a Polars DataFrame or LazyFrame
    containing GNSS observations.

    Args:
        df (pl.DataFrame | pl.LazyFrame): Input DataFrame or LazyFrame containing GNSS
            observations.
        sampling_interval (int | None, optional): Sampling interval in seconds. If
            None, it will be inferred from the data. Defaults to None.
        bias_fn (str | Path | Iterable[str | Path] | None, optional): Path(s) to the
            bias file(s). If provided, DCB biases will be applied to the TEC
            calculation. Defaults to None.
        config (TECConfig, optional): Configuration parameters for TEC calculation.
            Defaults to TECConfig().
        rx_bias (bool, optional): Whether to apply receiver bias correction. If False,
            mapping function column will be retained for possible subsequent receiver
            bias correction. Default is True.

    Returns:
        pl.LazyFrame: A LazyFrame containing the calculated TEC values.
    """
    lf = (
        df.lazy()
        # Filter by minimum elevation angle and constellations
        .filter(
            pl.col("elevation") >= config.min_elevation,
            pl.col("prn").cat.slice(0, 1).is_in(list(config.constellations)),
        )
        # drop D-code and S-code observations
        .drop(cs.matches(r"^[D,S]\d[A-Z]$"))
    )

    if sampling_interval is None:
        sampling_interval = int(
            lf.select(
                pl.col("time")
                .diff()
                .min()
                .over("station", "prn")
                .mean()
                .dt.total_seconds()
            )
            .collect()
            .item()
        )

    sampling_config = get_sampling_config(sampling_interval)

    # Determine if the time column is in UTC or GPS time
    is_utc = lf.head(1).collect().get_column("time")[0].tzinfo.key == "UTC"
    leap_seconds = (
        get_leap_seconds(pl.col("time").dt.replace_time_zone(None))
        if is_utc
        else pl.duration(seconds=0)
    )

    lf = lf.with_columns(
        # Adjust time to GPS time for correctly joining with bias data
        pl.col("time").add(leap_seconds).dt.replace_time_zone(None)
    )
    bias_lf = None if bias_fn is None else read_bias(bias_fn)
    lf = _coalesce_observations(lf, bias_lf, rx_bias, config)
    lf = _map_frequencies(lf)

    f1 = pl.col("C1_freq")
    f2 = pl.col("C2_freq")
    coeff = f1**2 * f2**2 / (f1**2 - f2**2) / 40.3e16
    tecu_per_ns = 1e-9 * c * coeff

    lf = (
        lf.drop_nulls(["C1", "C2", "L1", "L2", "C1_freq", "C2_freq"])
        .with_columns(
            # sTEC from pseudorange, in TECU
            (pl.col("C2") - pl.col("C1")).mul(coeff).alias("stec_g"),
            # sTEC from carrier phase, in TECU
            (pl.col("L1") / f1 - pl.col("L2") / f2).mul(c * coeff).alias("stec_p"),
        )
        .drop("C1", "C2", "L1", "L2")
        # Identify arcs based on time gaps
        .with_columns(
            pl.col("time")
            .diff()
            .ge(sampling_config.arc_interval)
            .fill_null(False)
            .cum_sum()
            .cast(pl.UInt16)
            .over("station", "prn")
            .alias("arc_id")
        )
        # Detect and correct cycle slips in each arc
        .with_columns(
            pl.struct([pl.col("time").dt.epoch("s"), pl.col("stec_p")])
            .map_batches(
                lambda x: _correct_cycle_slip(
                    x.struct.field("time").to_numpy(),
                    x.struct.field("stec_p").to_numpy(),
                    sampling_config.slip_tec_threshold,
                    sampling_config.slip_correction_window,
                ),
                return_dtype=pl.Float64,
            )
            .over("station", "prn", "arc_id")
            .alias("stec_p")
        )
        # Level phase sTEC to pseudorange sTEC using elevation-based weighted offset
        .with_columns(
            (pl.col("stec_g") - pl.col("stec_p")).alias("raw_offset"),
            pl.col("elevation").radians().sin().pow(2).alias("weight"),
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
        .drop("raw_offset", "weight", "offset")
    )

    intermediate_cols = {"azimuth", "elevation", "stec_g", "stec_p", "arc_id"}
    if bias_fn is not None:
        intermediate_cols.update({"tx_bias", "rx_bias"})
        lf = lf.with_columns(
            # Convert biases from ns to TECU
            pl.col("tx_bias").mul(tecu_per_ns),
            pl.col("rx_bias").mul(tecu_per_ns),
        ).with_columns(
            # sTEC corrected for DCB biases, in TECU
            pl.col("stec").sub(pl.col("tx_bias") + pl.col("rx_bias").fill_null(0))
        )

    mf, ipp_lat, ipp_lon = single_layer_model(
        pl.col("azimuth"),
        pl.col("elevation"),
        pl.col("rx_lat"),
        pl.col("rx_lon"),
        config,
    )

    lf = (
        lf.with_columns(
            # vTEC, in TECU
            (pl.col("stec") / mf).alias("vtec"),
            # IPP Latitude in degrees
            ipp_lat.alias("ipp_lat"),
            # IPP Longitude in degrees
            ipp_lon.alias("ipp_lon"),
            # Adjust time back to UTC
            pl.col("time").sub(leap_seconds).dt.replace_time_zone("UTC"),
        )
        .drop("C1_freq", "C2_freq")
        .sort("time", "station", "prn")
    )

    if not rx_bias:
        lf = lf.with_columns(mf.alias("mf"))

    if config.retain_intermediate != "all":
        cols_to_retain = set()
        if config.retain_intermediate is None:
            pass
        elif isinstance(config.retain_intermediate, str):
            cols_to_retain.add(config.retain_intermediate)
        else:
            cols_to_retain.update(config.retain_intermediate)

        cols_available = set(lf.collect_schema().names())
        cols_to_drop = cols_available.intersection(intermediate_cols) - cols_to_retain
        lf = lf.drop(cols_to_drop)

    return lf


def calc_tec_from_parquet(
    parquet_fn: str | Path,
    bias_fn: str | Path | Iterable[str | Path] | None = None,
    config: TECConfig = TECConfig(),
    *,
    rx_bias: bool = True,
) -> pl.LazyFrame:
    """
    Calculate the Total Electron Content (TEC) from a Parquet file containing GNSS
    observations. Make sure the Parquet file contains necessary metadata such as
    sampling interval and receiver geodetic coordinates.

    Args:
        parquet_fn (str | Path): Path to the Parquet file.
        bias_fn (str | Path | Iterable[str | Path] | None, optional): Path(s) to the
            bias file(s). If provided, DCB biases will be applied to the TEC
            calculation. Defaults to None.
        config (TECConfig, optional): Configuration parameters for TEC calculation.
            Defaults to TECConfig().
        rx_bias (bool, optional): Whether to apply receiver bias correction. If False,
            mapping function column will be retained for possible subsequent receiver
            bias correction. Default is True.

    Returns:
        pl.LazyFrame: A LazyFrame containing the calculated TEC values.
    """
    lf = pl.scan_parquet(parquet_fn)
    metadata = pl.read_parquet_metadata(parquet_fn)

    sampling_interval = metadata.get("sampling_interval", None)
    if sampling_interval is not None:
        sampling_interval = int(sampling_interval)

    rx_lat = metadata.get("rx_geodetic_lat", None)
    if rx_lat is not None:
        rx_lat = float(rx_lat)
    else:
        raise ValueError("Receiver latitude not found in parquet metadata.")

    rx_lon = metadata.get("rx_geodetic_lon", None)
    if rx_lon is not None:
        rx_lon = float(rx_lon)
    else:
        raise ValueError("Receiver longitude not found in parquet metadata.")

    lf = lf.with_columns(
        pl.lit(rx_lat, dtype=pl.Float32).alias("rx_lat"),
        pl.lit(rx_lon, dtype=pl.Float32).alias("rx_lon"),
    )

    return calc_tec_from_df(lf, sampling_interval, bias_fn, config, rx_bias=rx_bias)


def calc_tec_from_rinex(
    obs_fn: str | Path | Iterable[str | Path],
    nav_fn: str | Path | Iterable[str | Path],
    bias_fn: str | Path | Iterable[str | Path] | None = None,
    config: TECConfig = TECConfig(),
    *,
    station: str | None = None,
    rx_bias: bool = True,
) -> pl.LazyFrame:
    """
    Calculate the Total Electron Content (TEC) from RINEX observation and navigation
    files.

    Args:
        obs_fn (str | Path | Iterable[str | Path]): Path(s) to the RINEX observation
            file(s). These files must be from the same station, otherwise the output
            DataFrame will be incorrect.
        nav_fn (str | Path | Iterable[str | Path]): Path(s) to the RINEX navigation
            file(s).
        bias_fn (str | Path | Iterable[str | Path] | None, optional): Path(s) to the
            bias file(s). If provided, DCB biases will be applied to the TEC
            calculation. Defaults to None.
        config (TECConfig, optional): Configuration parameters for TEC calculation.
            Defaults to TECConfig().
        station (str | None, optional): Custom station name to assign to the data. If
            None, the station name from the RINEX header is used. Defaults to None.
        rx_bias (bool, optional): Whether to apply receiver bias correction. If False,
            mapping function column will be retained for possible subsequent receiver
            bias correction. Default is True.

    Returns:
        pl.LazyFrame: A LazyFrame containing the calculated TEC values.
    """
    header, lf = read_rinex_obs(obs_fn, nav_fn, config.constellations, station=station)

    lf = lf.with_columns(
        pl.lit(header.rx_geodetic[0], dtype=pl.Float32).alias("rx_lat"),
        pl.lit(header.rx_geodetic[1], dtype=pl.Float32).alias("rx_lon"),
    )

    return calc_tec_from_df(
        lf, header.sampling_interval, bias_fn, config, rx_bias=rx_bias
    )
