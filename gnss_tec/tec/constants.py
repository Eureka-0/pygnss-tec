from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Literal

import polars as pl

c = 299792458.0
"""Speed of light in m/s"""

Re = 6378.137e3
"""Earth radius in meters."""

SUPPORTED_RINEX_VERSIONS = ["3"]
"""Supported RINEX major versions."""

SUPPORTED_CONSTELLATIONS = {"C": "BDS", "G": "GPS"}
"""Supported GNSS constellations for TEC calculation."""

SIGNAL_FREQ = {
    "C": {
        "B1-2": 1561.098e6,
        "B1": 1575.42e6,
        "B2a": 1176.45e6,
        "B2b": 1207.14e6,
        "B2": 1191.795e6,
        "B3": 1268.52e6,
    },
    "G": {"L1": 1575.42e6, "L2": 1227.60e6, "L5": 1176.45e6},
}
"""Signal frequencies for supported constellations and signals in Hz."""

DEFAULT_C1_CODES = {
    "C": ["C2I", "C2D", "C2X", "C1I", "C1D", "C1X", "C2W", "C1C"],
    "G": ["C1W", "C1C", "C1X"],
}

DEFAULT_C2_CODES = {
    "C": ["C6I", "C6D", "C6X", "C7I", "C7D", "C7X", "C5I", "C5D", "C5X"],
    "G": ["C2W", "C2C", "C2X", "C5W", "C5C", "C5X"],
}


@dataclass(frozen=True, kw_only=True)
class TECConfig:
    constellations: str = field(
        default_factory=lambda: "".join(SUPPORTED_CONSTELLATIONS.keys())
    )
    """Constellations to consider for TEC calculation."""

    ipp_height: float = 400
    """Ionospheric pierce point height in kilometers."""

    min_elevation: float = 30.0
    """Minimum satellite elevation angle in degrees."""

    min_snr: float = 30.0
    """Minimum signal-to-noise ratio in dB-Hz."""

    c1_codes: Mapping[str, list[str]] = field(default_factory=lambda: {})
    """Observation codes priority list for C1 measurements."""

    c2_codes: Mapping[str, list[str]] = field(default_factory=lambda: {})
    """Observation codes priority list for C2 measurements."""

    rx_bias: Literal["external", "mstd", "lsq"] | None = "external"
    """Method to correct receiver bias.
        - "external": Use external bias file(s). If the station is not found in the bias
            file(s), this will result in an empty dataframe.
        - "mstd": Estimate receiver bias using the Minimum Standard Deviation method.
        - "lsq": Estimate receiver bias using Least Squares fitting.
        - None: Do not correct receiver bias.
    """

    mapping_function: Literal["slm", "mslm"] = "slm"
    """Mapping function to use:
        - "slm": Single Layer Model
        - "mslm": Modified Single Layer Model
    """

    retain_intermediate: str | Iterable[str] | None | Literal["all"] = None
    """Names of intermediate columns to retain in the output DataFrame."""

    @property
    def ipp_height_m(self) -> float:
        """Ionospheric pierce point height in meters."""
        return self.ipp_height * 1e3

    @property
    def mslm_height_m(self) -> float:
        """Ionospheric pierce point height for Modified Single Layer Model in meters."""
        return 506.7e3

    @property
    def alpha(self) -> float:
        """Correction factor for Modified Single Layer Model."""
        return 0.9782

    @property
    def code2band(self) -> Mapping[str, int]:
        code_band: dict[str, int] = {}
        for const, code, _ in self.iter_c1_codes():
            code_band[f"{const}_{code}"] = 1  # C1 band
        for const, code, _ in self.iter_c2_codes():
            code_band[f"{const}_{code}"] = 2  # C2 band
        return code_band

    @property
    def c1_priority(self) -> Mapping[str, int]:
        priority: dict[str, int] = {}
        for const, code, i in self.iter_c1_codes():
            priority[f"{const}_{code}"] = i
        return priority

    @property
    def c2_priority(self) -> Mapping[str, int]:
        priority: dict[str, int] = {}
        for const, code, i in self.iter_c2_codes():
            priority[f"{const}_{code}"] = i
        return priority

    def iter_c1_codes(self) -> Iterator[tuple[str, str, int]]:
        for constellation, codes in self.c1_codes.items():
            for i, code in enumerate(codes):
                yield constellation, code, i

    def iter_c2_codes(self) -> Iterator[tuple[str, str, int]]:
        for constellation, codes in self.c2_codes.items():
            for i, code in enumerate(codes):
                yield constellation, code, i

    def __post_init__(self):
        # Validate constellations
        allowed = set(SUPPORTED_CONSTELLATIONS.keys())
        actual = set(self.constellations)

        invalid = actual - allowed
        if invalid:
            raise ValueError(
                f"Invalid constellations {self.constellations!r}; "
                f"allowed letters are subset of {''.join(sorted(allowed))!r}."
            )

        # Set default codes if not provided
        if not self.c1_codes:
            object.__setattr__(self, "c1_codes", DEFAULT_C1_CODES)
        else:
            user_codes = dict(self.c1_codes)
            unknown = user_codes.keys() - allowed
            if unknown:
                raise ValueError(
                    f"Invalid constellations in c1_codes: {unknown}. "
                    f"Allowed constellations are {allowed}."
                )
            object.__setattr__(self, "c1_codes", DEFAULT_C1_CODES | user_codes)

        if not self.c2_codes:
            object.__setattr__(self, "c2_codes", DEFAULT_C2_CODES)
        else:
            user_codes = dict(self.c2_codes)
            unknown = user_codes.keys() - allowed
            if unknown:
                raise ValueError(
                    f"Invalid constellations in c2_codes: {unknown}. "
                    f"Allowed constellations are {allowed}."
                )
            object.__setattr__(self, "c2_codes", DEFAULT_C2_CODES | user_codes)


@dataclass(frozen=True)
class SamplingConfig:
    arc_interval: pl.Expr
    """Minimum time interval for arc segmentation (pl.duration)."""

    slip_tec_threshold: float
    """TECU threshold to detect cycle slips."""

    slip_correction_window: int
    """Window size for slip correction in number of samples."""


def get_sampling_config(sampling_interval: int) -> SamplingConfig:
    if sampling_interval <= 5:
        return SamplingConfig(
            arc_interval=pl.duration(minutes=1),
            slip_tec_threshold=0.5,
            slip_correction_window=20,
        )
    else:
        return SamplingConfig(
            arc_interval=pl.duration(minutes=5),
            slip_tec_threshold=2.0,
            slip_correction_window=10,
        )
