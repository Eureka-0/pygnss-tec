import gzip
import io
from pathlib import Path
from typing import Iterable, Literal, overload

import pandas as pd
import polars as pl


def _read_bias_file(fn: str | Path) -> pl.LazyFrame:
    with gzip.open(fn, "rt") as f:
        for line in f:
            if "+BIAS/SOLUTION" in line:
                header_str = next(f).rstrip("\n")
                break
        else:
            raise ValueError("Header '+BIAS/SOLUTION' not found in file.")

        data_lines = []
        for line in f:
            if "-BIAS/SOLUTION" in line:
                break
            data_lines.append(line)

    header_list = header_str.split()
    cols = [col.strip("*_") for col in header_list]
    colspecs = []
    for col in header_list:
        start_idx = header_str.index(col)
        end_idx = start_idx + len(col)
        colspecs.append((start_idx, end_idx))

    buf = io.StringIO("".join(data_lines))
    df = (
        pl.from_pandas(pd.read_fwf(buf, colspecs=colspecs, names=cols, header=None))
        .lazy()
        .drop_nulls("UNIT")
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

    df_list = [_read_bias_file(f) for f in fn_list]
    df = pl.concat(df_list)

    if lazy:
        return df
    else:
        return df.collect()
