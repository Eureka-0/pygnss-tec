from __future__ import annotations

import gzip
import io
import re
from pathlib import Path
from typing import Iterable, Literal, overload

import pandas as pd
import polars as pl


def _read_bias_file(fn: str | Path) -> pl.LazyFrame:
    with gzip.open(fn, "rt") as f:
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
    cols = [col.strip("*_") for col in header_str.split()]
    df = (
        pl.from_pandas(pd.read_fwf(buf, colspecs=colspecs, names=cols, header=None))
        .lazy()
        .drop("BIAS", "SVN")
    )

    for col in ["BIAS_START", "BIAS_END"]:
        df = (
            df.with_columns(pl.col(col).str.split(":").alias("parts"))
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

    return df


@overload
def read_bias(
    fn: str | Path | Iterable[str | Path], *, lazy: Literal[True]
) -> pl.LazyFrame: ...


@overload
def read_bias(
    fn: str | Path | Iterable[str | Path], *, lazy: Literal[False] = False
) -> pl.DataFrame: ...


def read_bias(
    fn: str | Path | Iterable[str | Path], *, lazy: bool = False
) -> pl.DataFrame | pl.LazyFrame:
    """Read GNSS DCB bias files into a Polars DataFrame.
    Args:
        fn (str | Path | Iterable[str | Path]): Path(s) to the bias file(s).
        lazy (bool, optional): Whether to return a lazy DataFrame. Defaults to False.

    Returns:
        (pl.DataFrame | pl.LazyFrame): A DataFrame or LazyFrame containing the bias
            data.
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

    df = pl.concat([_read_bias_file(f) for f in fn_list])

    if lazy:
        return df
    else:
        return df.collect()
