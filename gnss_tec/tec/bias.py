from __future__ import annotations

import gzip
import io
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

import polars as pl
from scipy.optimize import minimize_scalar

from ..rinex import get_leap_seconds


def _read_bias_file(fn: str | Path) -> pl.LazyFrame:
    if str(fn).endswith(".gz"):
        with gzip.open(fn, "rt") as f:
            lines = f.readlines()
    else:
        with open(fn, "r") as f:
            lines = f.readlines()

    if not lines:
        raise ValueError(f"Bias file {fn} is empty.")

    try:
        header_marker_idx = next(
            i for i, line in enumerate(lines) if "+BIAS/SOLUTION" in line
        )
    except StopIteration:
        raise ValueError("Header '+BIAS/SOLUTION' not found in the file.")

    header_line_idx = header_marker_idx + 1
    if header_line_idx >= len(lines):
        raise ValueError("No header line found after '+BIAS/SOLUTION' marker.")

    header_str = lines[header_line_idx].rstrip("\n")

    footer_line_idx = None
    for i in range(len(lines) - 1, header_line_idx, -1):
        if "-BIAS/SOLUTION" in lines[i]:
            footer_line_idx = i
            break

    if footer_line_idx is None:
        footer_line_idx = len(lines)

    if footer_line_idx <= header_line_idx + 1:
        buf = io.StringIO("")
    else:
        buf = io.StringIO("".join(lines[header_line_idx + 1 : footer_line_idx]))

    colspecs = [(m.start(), m.end()) for m in re.finditer(r"\S+", header_str)]
    cols = [col.strip("*_").lower() for col in header_str.split()]
    schema = {
        "prn": pl.Categorical,
        "station": pl.Categorical,
        "obs1": pl.Categorical,
        "obs2": pl.Categorical,
        "unit": pl.Categorical,
        "estimated_value": pl.Float64,
        "std_dev": pl.Float64,
    }
    lf = (
        pl.scan_csv(buf, has_header=False, new_columns=["full_str"])
        .with_columns(
            [
                pl.col("full_str")
                .str.slice(colspec[0], colspec[1] - colspec[0])
                .str.strip_chars()
                .replace("", None)
                .cast(schema.get(col, pl.String))
                .alias(col)
                for colspec, col in zip(colspecs, cols)
            ]
        )
        .drop("full_str", "bias", "svn")
    )

    for col in ["bias_start", "bias_end"]:
        lf = (
            lf.with_columns(pl.col(col).str.split(":").alias("parts"))
            .with_columns(
                pl.col("parts").list.get(0).alias("year"),
                pl.col("parts").list.get(1).alias("doy"),
                pl.col("parts").list.get(2).cast(pl.Int64).alias("sod"),
            )
            .with_columns(
                pl.concat_str(pl.col("year"), pl.lit("-"), pl.col("doy"))
                .str.strptime(pl.Date, "%Y-%j")
                .cast(pl.Datetime)
                .add(pl.col("sod") * pl.duration(seconds=1))
                .alias(col)
            )
            .drop("parts", "year", "doy", "sod")
        )

    return lf


def read_bias(fn: str | Path | Iterable[str | Path]) -> pl.LazyFrame:
    """
    Read GNSS DCB bias files into a Polars DataFrame.

    Args:
        fn (str | Path | Iterable[str | Path]): Path(s) to the bias file(s).

    Returns:
        pl.LazyFrame: A LazyFrame containing the bias data.
    """
    if isinstance(fn, (str, Path)):
        fn_list = [str(fn)]
    elif isinstance(fn, Iterable):
        fn_list = [str(f) for f in fn]
    else:
        raise TypeError("fn must be a str, Path, or Iterable of str/Path.")

    for f in fn_list:
        if not Path(f).exists():
            raise FileNotFoundError(f"Bias file not found: {f}")

    return pl.concat([_read_bias_file(f) for f in fn_list])


def correct_rx_bias(
    df: pl.DataFrame | pl.LazyFrame,
    method: Literal["mstd"] = "mstd",
    retain_rx_bias: bool = False,
) -> pl.LazyFrame:
    """
    Correct receiver bias in sTEC measurements using the specified method.

    Args:
        df (pl.DataFrame | pl.LazyFrame): Input DataFrame containing sTEC measurements.
        method (Literal["mstd"], optional): Method for bias correction. Defaults to "mstd".
        retain_rx_bias (bool, optional): Whether to retain the estimated receiver bias
            in the output DataFrame. Defaults to False.

    Returns:
        pl.LazyFrame: LazyFrame with receiver bias corrected sTEC and computed vTEC.
    """
    if method == "mstd":
        estimate_func = _mstd_rx_bias
    else:
        raise ValueError(f"Unknown bias correction method: {method}")

    lf = df.lazy()

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

    lf = lf.with_columns(
        pl.col("time").dt.date().alias("date"),
        pl.col("prn").cat.slice(0, 1).alias("constellation"),
        pl.col("time").add(pl.duration(hours=pl.col("ipp_lon") / 15)).alias("lt"),
    ).with_columns(
        pl.col("lt").sub(pl.col("lt").dt.truncate("1d")).dt.total_hours(fractional=True)
    )

    schema = lf.collect_schema()
    schema.update({"rx_bias": pl.Float64()})
    lf = (
        lf.group_by("date", "station", "constellation", "C1_code", "C2_code")
        .map_groups(estimate_func, schema=schema)
        .with_columns(pl.col("stec").sub(pl.col("rx_bias").fill_null(0)))
        .with_columns(
            pl.col("stec").truediv(pl.col("mf")).alias("vtec"),
            # Adjust time back to UTC
            pl.col("time").sub(leap_seconds).dt.replace_time_zone("UTC"),
        )
        .drop("date", "constellation", "lt", "mf")
    )

    if not retain_rx_bias:
        lf = lf.drop("rx_bias")

    return lf


def _mstd_rx_bias(df: pl.DataFrame) -> pl.DataFrame:
    df_night = df.filter((pl.col("lt") >= 18) | (pl.col("lt") <= 6))
    if df_night.height < 10:
        return df.with_columns(pl.lit(None).alias("rx_bias"))

    def mean_std(bias: float) -> float:
        corrected = df_night.with_columns(
            (pl.col("stec").sub(bias) / pl.col("mf")).alias("vtec")
        ).with_columns(pl.col("vtec").std().over("time").mean().alias("mean_std"))

        if corrected.filter(pl.col("vtec") <= 0).height > 0:
            return 1e6
        else:
            return corrected.get_column("mean_std").item(0)

    result = minimize_scalar(mean_std, bounds=(-500, 500), method="bounded")
    return df.with_columns(pl.lit(result.x).alias("rx_bias"))
