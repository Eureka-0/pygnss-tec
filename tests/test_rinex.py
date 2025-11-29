# from polars.testing import assert_frame_equal

from gnss_tec import read_rinex_obs


def test_read_rinex_obs(rinex_obs_v3_hatanaka, rinex_obs_v3):
    header1, df_hatanaka = read_rinex_obs(rinex_obs_v3_hatanaka)
    header2, df = read_rinex_obs(rinex_obs_v3)

    assert header1 == header2
    # assert_frame_equal(df_hatanaka, df, check_exact=False, abs_tol=1e-8)
    assert df.shape[0] > 0 and df.shape[1] > 0
    assert "time" in df.columns and "prn" in df.columns
