from polars.testing import assert_frame_equal

import gnss_tec as gt


def test_calc_tec_from_rinex(rinex_obs_v3_hatanaka, rinex_obs_v3, rinex_nav_v3, bias):
    df = gt.calc_tec_from_rinex(rinex_obs_v3, rinex_nav_v3, bias).collect()
    df_hatanaka = gt.calc_tec_from_rinex(
        rinex_obs_v3_hatanaka, rinex_nav_v3, bias
    ).collect()

    assert_frame_equal(df_hatanaka, df, check_exact=False, abs_tol=1e-8)
    assert df.shape[0] > 0 and df.shape[1] > 0
    assert "time" in df.columns
    assert "station" in df.columns
    assert "prn" in df.columns
    assert "ipp_lat" in df.columns
    assert "ipp_lon" in df.columns
    assert "stec" in df.columns
    assert "stec_dcb_corrected" in df.columns
    assert "vtec" in df.columns


def test_calc_tec_from_df(rinex_obs_v3, rinex_nav_v3, bias):
    header, lf = gt.read_rinex_obs(rinex_obs_v3, rinex_nav_v3)
    df = gt.calc_tec_from_df(lf, header, bias).collect()

    assert df.shape[0] > 0 and df.shape[1] > 0
    assert "time" in df.columns
    assert "station" in df.columns
    assert "prn" in df.columns
    assert "ipp_lat" in df.columns
    assert "ipp_lon" in df.columns
    assert "stec" in df.columns
    assert "stec_dcb_corrected" in df.columns
    assert "vtec" in df.columns


def test_mstd_bias(rinex_obs_v3, rinex_nav_v3, bias):
    df = gt.calc_tec_from_rinex(
        rinex_obs_v3, rinex_nav_v3, bias, gt.TECConfig(rx_bias="mstd")
    ).collect()

    assert df.shape[0] > 0 and df.shape[1] > 0
    assert "time" in df.columns
    assert "station" in df.columns
    assert "prn" in df.columns
    assert "ipp_lat" in df.columns
    assert "ipp_lon" in df.columns
    assert "stec" in df.columns
    assert "stec_dcb_corrected" in df.columns
    assert "vtec" in df.columns


def test_lsq_bias(rinex_obs_v3, rinex_nav_v3, bias):
    df = gt.calc_tec_from_rinex(
        rinex_obs_v3, rinex_nav_v3, bias, gt.TECConfig(rx_bias="lsq")
    ).collect()

    assert df.shape[0] > 0 and df.shape[1] > 0
    assert "time" in df.columns
    assert "station" in df.columns
    assert "prn" in df.columns
    assert "ipp_lat" in df.columns
    assert "ipp_lon" in df.columns
    assert "stec" in df.columns
    assert "stec_dcb_corrected" in df.columns
    assert "vtec" in df.columns
