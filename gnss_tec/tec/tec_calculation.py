from __future__ import annotations

from pathlib import Path
from typing import Iterable

import polars as pl
import polars.selectors as cs

from .constants import C1_CODES, C2_CODES, SIGNAL_FREQ, SUPPORTED_CONSTELLATIONS

CODE_FREQ_MAP = {
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


def _resolve_observations(df: pl.LazyFrame) -> pl.LazyFrame:
    columns = df.collect_schema().names()
    valid_c1_codes = {
        constellation: list(filter(lambda c: c in columns, codes))
        for constellation, codes in C1_CODES.items()
    }
    valid_c2_codes = {
        constellation: list(filter(lambda c: c in columns, codes))
        for constellation, codes in C2_CODES.items()
    }
    valid_l1_codes = {
        constellation: [f"L{c[1:]}" for c in codes]
        for constellation, codes in valid_c1_codes.items()
    }
    valid_l2_codes = {
        constellation: [f"L{c[1:]}" for c in codes]
        for constellation, codes in valid_c2_codes.items()
    }

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

    return (
        df.with_columns(
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


def _map_frequencies(df: pl.LazyFrame) -> pl.LazyFrame:
    def freq_col(code_col: str) -> pl.Expr:
        expr = pl.when(False).then(None)
        for constellation in SUPPORTED_CONSTELLATIONS:
            expr = expr.when(pl.col("PRN").str.starts_with(constellation)).then(
                pl.col(code_col)
                .str.slice(1, 1)
                .str.pad_start(2, "_")
                .str.pad_start(3, constellation)
                .replace_strict(CODE_FREQ_MAP, default=None)
            )
        return expr

    return df.with_columns(
        freq_col("C1_Code").alias("C1_Freq"),
        freq_col("C2_Code").alias("C2_Freq"),
        freq_col("L1_Code").alias("L1_Freq"),
        freq_col("L2_Code").alias("L2_Freq"),
    ).drop("C1_Code", "C2_Code", "L1_Code", "L2_Code")


def calc_tec(
    obs_fn: str | Path | Iterable[str | Path],
    nav_fn: str | Path | Iterable[str | Path],
    constellations: str | None = None,
    t_lim: tuple[str | None, str | None] | list[str | None] | None = None,
): ...
