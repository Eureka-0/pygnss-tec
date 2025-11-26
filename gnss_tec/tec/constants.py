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

C1_CODES = {"C": ["C2I", "C2X", "C1X", "C2W", "C1C"], "G": ["C1W", "C1C", "C1X"]}
"""Observation codes priority list for C1 measurements."""

C2_CODES = {"C": ["C6I", "C7I", "C7D", "C5X"], "G": ["C2W", "C2X"]}
"""Observation codes priority list for C2 measurements."""

DEFAULT_IPP_HEIGHT = 400e3
"""Default ionospheric pierce point height in meters."""
