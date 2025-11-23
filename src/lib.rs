use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rinex::navigation::{Ephemeris, Perturbations};
use rinex::prelude::{qc::Merge, *};
use rustc_hash::{FxHashMap, FxHashSet};
use std::str::FromStr;

// All supported constellations: G - GPS, C - BeiDou, E - Galileo, R - GLONASS, J - QZSS, I - IRNSS, S - SBAS
const ALL_CONSTELLATIONS: &str = "GCERJIS";

fn read_rinex_file(path: &str) -> PyResult<Rinex> {
    if path.ends_with(".gz") {
        Rinex::from_gzip_file(path).map_err(|e| PyIOError::new_err(e.to_string()))
    } else {
        Rinex::from_file(path).map_err(|e| PyIOError::new_err(e.to_string()))
    }
}

fn read_rinex_files(paths: Vec<String>) -> PyResult<Rinex> {
    let first_path = paths
        .first()
        .ok_or_else(|| PyValueError::new_err("No RINEX file paths provided"))?;
    let mut rinex = read_rinex_file(first_path)?;
    for path in paths.iter().skip(1) {
        let next_rinex = read_rinex_file(path)?;
        let _ = rinex.merge_mut(&next_rinex);
    }
    Ok(rinex)
}

fn epoch_from_str(s: Option<&str>) -> Option<Epoch> {
    s.and_then(|t_str| Epoch::from_str(t_str).ok())
}

fn replenish_perturbations(eph: &Ephemeris) -> Ephemeris {
    let crc = eph.get_orbit_f64("crc");
    let crs = eph.get_orbit_f64("crs");

    if crc.is_some() && crs.is_some() {
        return eph.with_perturbations(Perturbations {
            cuc: eph.get_orbit_f64("cuc").unwrap_or(0.0),
            cus: eph.get_orbit_f64("cus").unwrap_or(0.0),
            cic: eph.get_orbit_f64("cic").unwrap_or(0.0),
            cis: eph.get_orbit_f64("cis").unwrap_or(0.0),
            crc: crc.unwrap(),
            crs: crs.unwrap(),
            dn: eph.get_orbit_f64("deltaN").unwrap_or(0.0),
            i_dot: eph.get_orbit_f64("idot").unwrap_or(0.0),
            omega_dot: eph.get_orbit_f64("omegaDot").unwrap_or(0.0),
        });
    } else {
        eph.clone()
    }
}

fn get_nav_pos(
    nav_rnx: &Rinex,
    epochs: Vec<Epoch>,
    svs: Vec<SV>,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let sv_set: FxHashSet<SV> = svs.iter().copied().collect();
    let mut ephs_by_sv: FxHashMap<SV, Vec<(Epoch, Ephemeris)>> = FxHashMap::default();
    for (key, eph) in nav_rnx.nav_ephemeris_frames_iter() {
        if sv_set.contains(&key.sv) {
            ephs_by_sv
                .entry(key.sv)
                .or_default()
                .push((key.epoch, replenish_perturbations(eph)));
        }
    }

    let len = epochs.len();
    let mut xs = Vec::with_capacity(len);
    let mut ys = Vec::with_capacity(len);
    let mut zs = Vec::with_capacity(len);
    epochs
        .into_iter()
        .zip(svs.into_iter())
        .for_each(|(epoch, sv)| {
            let pv = ephs_by_sv.get(&sv).and_then(|eph_list| {
                eph_list
                    .iter()
                    .min_by_key(|(eph_epoch, _)| (epoch - *eph_epoch).abs())
                    .and_then(|(_, eph)| eph.kepler2position_velocity(sv, epoch))
            });
            match pv {
                Some(pv) => {
                    xs.push(pv.0.x * 1e3);
                    ys.push(pv.0.y * 1e3);
                    zs.push(pv.0.z * 1e3);
                }
                None => {
                    xs.push(f64::NAN);
                    ys.push(f64::NAN);
                    zs.push(f64::NAN);
                }
            }
        });

    (xs, ys, zs)
}

fn pivot_observations(
    obs_rnx: &Rinex,
    const_filter: &FxHashSet<Constellation>,
    t_lim: (Option<Epoch>, Option<Epoch>),
    codes: Option<&Vec<String>>,
) -> (Vec<Epoch>, Vec<SV>, Vec<String>, Vec<Vec<f64>>) {
    // 行索引：(epoch, sv) -> row_idx
    let mut row_index: FxHashMap<(Epoch, SV), usize> = FxHashMap::default();

    // 列索引：code -> col_idx
    let mut code_index: FxHashMap<String, usize> = FxHashMap::default();

    // 时间和 SV 列
    let mut epochs: Vec<Epoch> = Vec::new();
    let mut svs: Vec<SV> = Vec::new();

    // 每个 observable code 对应的一列, columns[col_idx][row_idx] = value
    let mut columns: Vec<Vec<f64>> = Vec::new();

    // 记录列名的顺序（和 columns 一一对应）
    let mut code_names: Vec<String> = Vec::new();

    obs_rnx
        .signal_observations_iter()
        .filter(|(key, signal)| {
            (t_lim.0.map_or(true, |t1| key.epoch >= t1))
                && (t_lim.1.map_or(true, |t2| key.epoch <= t2))
                && const_filter.contains(&signal.sv.constellation)
                && codes.map_or(true, |c| c.contains(&signal.observable.to_string()))
        })
        .for_each(|(key, signal)| {
            // 行索引：确定 row_idx
            let row_key = (key.epoch, signal.sv);
            let row_idx = *row_index.entry(row_key).or_insert_with(|| {
                // 新行索引
                let idx = epochs.len();

                // 记录行的 Time, PRN
                epochs.push(key.epoch);
                svs.push(signal.sv);

                // 对于已有的每一列，在新行位置补一个 NAN
                for col in columns.iter_mut() {
                    col.push(f64::NAN);
                }

                idx
            });

            // 列索引：确定 col_idx
            let code = signal.observable.to_string();
            let col_idx = match code_index.get(&code) {
                Some(&idx) => idx,
                None => {
                    // 新的 observable code -> 新列
                    let idx = code_names.len();
                    code_names.push(code.clone());
                    code_index.insert(code.clone(), idx);

                    // 新列需要为现有所有行补 NAN
                    let col = vec![f64::NAN; epochs.len()];
                    columns.push(col);

                    idx
                }
            };

            // 写入单元格
            columns[col_idx][row_idx] = signal.value;
        });

    (epochs, svs, code_names, columns)
}

/// Read RINEX observation file (and optional navigation file) and return a Polars DataFrame.
/// # Arguments
/// - `obs_fn` - Vector of paths to RINEX observation files.
/// - `nav_fn` - Optional vector of paths to RINEX navigation files.
/// - `constellations` - Optional string of constellation codes to filter (e.g., "CGE").
/// - `t_lim` - Tuple of optional start and end time strings for filtering epochs.
/// - `codes` - Optional vector of observable codes to include.
/// # Returns
/// - `PyResult<PyDict>` - Dictionary containing the observation data.
#[pyfunction]
fn _read_obs(
    py: Python<'_>,
    obs_fn: Vec<String>,
    nav_fn: Option<Vec<String>>,
    constellations: Option<String>,
    t_lim: (Option<String>, Option<String>),
    codes: Option<Vec<String>>,
) -> PyResult<Py<PyDict>> {
    // Read RINEX observation file (and navigation file if provided)
    let obs_rnx = read_rinex_files(obs_fn)?;
    let nav_rnx = match nav_fn {
        Some(nav_path) => Some(read_rinex_files(nav_path)?),
        None => None,
    };
    let const_filter: FxHashSet<Constellation> = constellations
        .unwrap_or(ALL_CONSTELLATIONS.to_string())
        .chars()
        .filter(|&c| ALL_CONSTELLATIONS.contains(c))
        .filter_map(|c| Constellation::from_str(&c.to_string()).ok())
        .collect();
    let t1 = epoch_from_str(t_lim.0.as_deref());
    let t2 = epoch_from_str(t_lim.1.as_deref());

    let header = &obs_rnx.header;
    let version = format!("{}.{:02}", header.version.major, header.version.minor);
    let constellation = header.constellation.and_then(|c| Some(c.to_string()));
    let sampling_interval = header
        .sampling_interval
        .and_then(|duration| Some(duration.to_seconds() as u32));
    let leap_seconds = header.leap.and_then(|leap| Some(leap.leap));
    let (x_m, y_m, z_m) = header.rx_position.unwrap_or((f64::NAN, f64::NAN, f64::NAN));

    let marker = header.geodetic_marker.as_ref();
    let marker_name = marker
        .and_then(|m| Some(m.name.clone()))
        .unwrap_or("Unknown".to_string());
    let marker_type = marker.and_then(|m| m.marker_type.and_then(|mt| Some(mt.to_string())));

    let (epochs, svs, codes, columns) =
        pivot_observations(&obs_rnx, &const_filter, (t1, t2), codes.as_ref());

    // construct result dictionary
    let result_dict = PyDict::new(py);
    result_dict.set_item("Version", version)?;
    result_dict.set_item("Constellation", constellation)?;
    result_dict.set_item("SamplingInterval", sampling_interval)?;
    result_dict.set_item("LeapSeconds", leap_seconds)?;
    result_dict.set_item("Station", marker_name)?;
    result_dict.set_item("MarkerType", marker_type)?;
    result_dict.set_item("RX_X", x_m)?;
    result_dict.set_item("RX_Y", y_m)?;
    result_dict.set_item("RX_Z", z_m)?;
    result_dict.set_item(
        "Time",
        epochs
            .iter()
            .map(|epoch| epoch.to_unix_milliseconds())
            .collect::<Vec<f64>>(),
    )?;
    result_dict.set_item(
        "PRN",
        svs.iter().map(|sv| sv.to_string()).collect::<Vec<String>>(),
    )?;
    for (code, col) in codes.iter().zip(columns.iter()) {
        result_dict.set_item(code, col.clone())?;
    }

    // If navigation RINEX is provided, calculate Azimuth and Elevation
    match nav_rnx {
        Some(nav_rnx) => {
            let (nav_x, nav_y, nav_z) = get_nav_pos(&nav_rnx, epochs, svs);
            result_dict.set_item("NAV_X", nav_x)?;
            result_dict.set_item("NAV_Y", nav_y)?;
            result_dict.set_item("NAV_Z", nav_z)?;
        }
        None => {}
    }

    Ok(result_dict.into())
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_read_obs, m)?)?;
    Ok(())
}
