from polars.testing import assert_frame_equal

import gnss_tec as gt


def test_read_rinex_obs_v2(rinex_obs_v2, rinex_nav_v2):
    header, lf = gt.read_rinex_obs(rinex_obs_v2, rinex_nav_v2)
    df = lf.collect()

    assert header.version.startswith("2.")
    assert header.sampling_interval == 15
    assert df.shape[0] > 0 and df.shape[1] > 0
    assert {
        "time",
        "station",
        "prn",
        "azimuth",
        "elevation",
        "C1",
        "L1",
        "S1",
    }.issubset(set(df.columns))


def test_read_rinex_obs_v3(rinex_obs_v3_hatanaka, rinex_obs_v3, rinex_nav_v3):
    header1, lf_hatanaka = gt.read_rinex_obs(rinex_obs_v3_hatanaka, rinex_nav_v3)
    header2, lf = gt.read_rinex_obs(rinex_obs_v3, rinex_nav_v3)
    df_hatanaka = lf_hatanaka.collect()
    df = lf.collect()

    assert header1 == header2
    assert header1.version.startswith("3.")
    assert header1.sampling_interval == 30
    assert_frame_equal(df_hatanaka, df, check_exact=False, abs_tol=1e-8)
    assert df.shape[0] > 0 and df.shape[1] > 0
    assert {
        "time",
        "station",
        "prn",
        "azimuth",
        "elevation",
        "C1C",
        "L1C",
        "S1C",
    }.issubset(set(df.columns))
