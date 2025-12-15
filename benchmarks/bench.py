import os
from pathlib import Path
from time import perf_counter

import pandas as pd

import gnss_tec as gt

# Constants
OBS_V2 = "data/rinex_obs_v2/dgar0100.24o.gz"
NAV_V2 = "data/rinex_nav_v2/brdc0100.24n.gz"
OBS_V3 = "data/rinex_obs_v3/CIBG00IDN_R_20240100000_01D_30S_MO.rnx.gz"
OBS_V3_CRX = "data/rinex_obs_v3/CIBG00IDN_R_20240100000_01D_30S_MO.crx.gz"
OBS_V3_CRX_SMALL = "data/rinex_obs_v3/BELE00BRA_R_20240100000_01D_30S_MO.crx.gz"
NAV_V3 = "data/rinex_nav_v3/BRDC00IGS_R_20240100000_01D_MN.rnx.gz"
BIAS = "data/bias/CAS0OPSRAP_20240100000_01D_01D_DCB.BIA.gz"
OUTPUT_FILE = "benchmarks/benchmark.md"


def benchmark_read_rinex(obs_path: str, nav_path: str) -> tuple[str, float]:
    """Benchmark reading RINEX files."""
    start_time = perf_counter()
    header, df = gt.read_rinex_obs(obs_path, nav_path)
    df.collect()
    end_time = perf_counter()

    if header.version.startswith("2"):
        version = "2"
    elif header.version.startswith("3"):
        version = "3"
    else:
        raise ValueError(f"Unknown RINEX version: {header.version}")

    file_size = Path(obs_path).stat().st_size / (1024 * 1024)  # in MB

    if "crx" in obs_path.lower():
        hatanaka = " Hatanaka-compressed"
    else:
        hatanaka = ""

    return (
        f"Read RINEX v{version} ({file_size:.2f} MB{hatanaka})",
        end_time - start_time,
    )


def benchmark_calc_tec(
    obs_path: str, nav_path: str, bias_path: str
) -> tuple[str, float]:
    """Benchmark TEC calculation from RINEX files."""
    start_time = perf_counter()
    header, lf = gt.read_rinex_obs(obs_path, nav_path)
    df = gt.calc_tec_from_df(lf, header, bias_path)
    df.collect()
    end_time = perf_counter()

    if header.version.startswith("2"):
        version = "2"
    elif header.version.startswith("3"):
        version = "3"
    else:
        raise ValueError(f"Unknown RINEX version: {header.version}")

    file_size = Path(obs_path).stat().st_size / (1024 * 1024)  # in MB

    if "crx" in obs_path.lower():
        hatanaka = " Hatanaka-compressed"
    else:
        hatanaka = ""

    return (
        f"Calculate TEC from RINEX v{version} ({file_size:.2f} MB{hatanaka})",
        end_time - start_time,
    )


def main():
    """Main function to run benchmarks and write results."""
    benchmarks = [
        benchmark_read_rinex(OBS_V2, NAV_V2),
        benchmark_read_rinex(OBS_V3, NAV_V3),
        benchmark_read_rinex(OBS_V3_CRX, NAV_V3),
        benchmark_read_rinex(OBS_V3_CRX_SMALL, NAV_V3),
        benchmark_calc_tec(OBS_V2, NAV_V2, BIAS),
        benchmark_calc_tec(OBS_V3, NAV_V3, BIAS),
        benchmark_calc_tec(OBS_V3_CRX, NAV_V3, BIAS),
        benchmark_calc_tec(OBS_V3_CRX_SMALL, NAV_V3, BIAS),
    ]

    # Create a DataFrame from the benchmark results
    results_df = pd.DataFrame(
        {
            "Task": [b[0] for b in benchmarks],
            "Time (s)": [f"{b[1]:.4f}" for b in benchmarks],
        }
    )

    # Convert DataFrame to Markdown table
    markdown_table = results_df.to_markdown(index=False)

    # Get system information
    cpu_count = os.cpu_count()
    system_info = f"M2 Pro {cpu_count}-Core CPU"

    # Write the Markdown table to the output file
    with Path(OUTPUT_FILE).open("w") as f:
        f.write(f"# Benchmark Results (on {system_info})\n\n")
        f.write(markdown_table)
        f.write("\n")

    print(f"Benchmark results written to {OUTPUT_FILE}")
    print(markdown_table)


if __name__ == "__main__":
    main()
