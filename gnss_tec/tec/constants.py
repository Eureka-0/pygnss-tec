from __future__ import annotations

from dataclasses import dataclass, field

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


@dataclass(frozen=True)
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

    c1_codes: dict[str, list[str]] = field(
        default_factory=lambda: {
            "C": ["C2I", "C2D", "C2X", "C1I", "C1D", "C1X", "C2W", "C1C"],
            "G": ["C1W", "C1C", "C1X"],
        }
    )
    """Observation codes priority list for C1 measurements."""

    c2_codes: dict[str, list[str]] = field(
        default_factory=lambda: {
            "C": ["C6I", "C6D", "C6X", "C7I", "C7D", "C7X", "C5I", "C5D", "C5X"],
            "G": ["C2W", "C2C", "C2X", "C5W", "C5C", "C5X"],
        }
    )
    """Observation codes priority list for C2 measurements."""

    @property
    def ipp_height_m(self) -> float:
        """Ionospheric pierce point height in meters."""
        return self.ipp_height * 1e3

    def __post_init__(self):
        allowed = set(SUPPORTED_CONSTELLATIONS.keys())
        actual = set(self.constellations)

        invalid = actual - allowed
        if invalid:
            raise ValueError(
                f"Invalid constellations {self.constellations!r}; "
                f"allowed letters are subset of {''.join(sorted(allowed))}, "
                f"but got invalid: {''.join(sorted(invalid))}"
            )


@dataclass(frozen=True)
class SamplingConfig:
    arc_interval: pl.Expr
    """Minimum time interval for arc segmentation (pl.duration)."""

    slip_tec_threshold: float
    """TECU threshold to detect cycle slips."""

    slip_correction_window: int
    """Window size for slip correction in number of samples (correct to window mean)."""


def get_sampling_config(sampling_interval: int) -> SamplingConfig:
    if sampling_interval <= 5:
        return SamplingConfig(
            arc_interval=pl.duration(minutes=1),
            slip_tec_threshold=1.0,
            slip_correction_window=20,
        )
    else:
        return SamplingConfig(
            arc_interval=pl.duration(minutes=5),
            slip_tec_threshold=5.0,
            slip_correction_window=10,
        )
