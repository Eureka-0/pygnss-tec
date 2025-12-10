from pathlib import Path

from pytest import fixture


@fixture
def test_data_dir():
    return Path(__file__).parent.parent / "data"


@fixture
def rinex_obs_v2(test_data_dir):
    return test_data_dir / "rinex_obs_v2/bele0100.24o.gz"


@fixture
def rinex_nav_v2(test_data_dir):
    return test_data_dir / "rinex_nav_v2/brdc0100.24g.gz"


@fixture
def rinex_obs_v3_hatanaka(test_data_dir):
    return test_data_dir / "rinex_obs_v3/CIBG00IDN_R_20240100000_01D_30S_MO.crx.gz"


@fixture
def rinex_obs_v3(test_data_dir):
    return test_data_dir / "rinex_obs_v3/CIBG00IDN_R_20240100000_01D_30S_MO.rnx.gz"


@fixture
def rinex_nav_v3(test_data_dir):
    return test_data_dir / "rinex_nav_v3/BRDC00IGS_R_20240100000_01D_MN.rnx.gz"


@fixture
def bias(test_data_dir):
    return test_data_dir / "bias/CAS0OPSRAP_20240100000_01D_01D_DCB.BIA.gz"
