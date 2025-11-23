import logging
from time import perf_counter

from polars.testing import assert_frame_equal

from gnss_tec import read_rinex_obs


def test_read_rinex_obs(rinex_obs_v3_hatanaka, rinex_obs_v3):
    start = perf_counter()
    header1, df_hatanaka = read_rinex_obs(rinex_obs_v3_hatanaka)
    end = perf_counter()
    logging.info("Time taken to read RINEX .crx.gz file: %.2f seconds", end - start)

    start = perf_counter()
    header2, df = read_rinex_obs(rinex_obs_v3)
    end = perf_counter()
    logging.info("Time taken to read RINEX .rnx file: %.2f seconds", end - start)

    assert header1 == header2
    assert_frame_equal(df_hatanaka, df, check_exact=False, abs_tol=1e-8)
    assert df.shape[0] > 0 and df.shape[1] > 0
    assert "Time" in df.columns and "PRN" in df.columns
