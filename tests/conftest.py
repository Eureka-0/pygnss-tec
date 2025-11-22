from pathlib import Path

from pytest import fixture


@fixture
def test_data_dir():
    return Path(__file__).parent.parent / "data"


@fixture
def rinex_obs_v3_hatanaka(test_data_dir):
    return test_data_dir / "rinex_obs_v3/CIBG00IDN_R_20240100000_01D_30S_MO.crx.gz"


@fixture
def rinex_obs_v3(test_data_dir):
    return test_data_dir / "rinex_obs_v3/CIBG00IDN_R_20240100000_01D_30S_MO.rnx.gz"
