use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use std::str::FromStr;

use arrow::array::{Array, ArrayData, make_array};
use arrow::array::{Float64Array, Int64Array, LargeStringArray};
use arrow::pyarrow::PyArrowType;
use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rinex::navigation::{Ephemeris, Perturbations};
use rinex::prelude::{qc::Merge, *};

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

/// Read RINEX observation file (and optional navigation file) and return a Polars DataFrame.
/// # Arguments
/// - `obs_fn` - String path to the RINEX observation file.
/// - `constellations` - Optional string of constellation codes to filter (e.g., "CGE").
/// - `include_doppler` - Boolean indicating whether to include Doppler observations.
/// # Returns
/// - `PyResult<PyDict>` - Dictionary containing the RINEX header information.
/// - `PyArrowType<ArrayData>` - Array of observation times in milliseconds since unix epoch.
/// - `PyArrowType<ArrayData>` - Array of PRN strings.
/// - `PyArrowType<ArrayData>` - Array of observation code strings.
/// - `PyArrowType<ArrayData>` - Array of observation values as floats.
#[pyfunction]
fn _read_obs(
    py: Python<'_>,
    obs_fn: String,
    constellations: Option<String>,
    include_doppler: bool,
) -> PyResult<(
    Py<PyDict>,
    PyArrowType<ArrayData>,
    PyArrowType<ArrayData>,
    PyArrowType<ArrayData>,
    PyArrowType<ArrayData>,
)> {
    let obs_rnx = read_rinex_file(&obs_fn)?;
    let const_filter: FxHashSet<Constellation> = constellations
        .unwrap_or(ALL_CONSTELLATIONS.to_string())
        .chars()
        .filter(|&c| ALL_CONSTELLATIONS.contains(c))
        .filter_map(|c| Constellation::from_str(&c.to_string()).ok())
        .collect();

    // construct header dict
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

    let header_dict = PyDict::new(py);
    header_dict.set_item("version", version)?;
    header_dict.set_item("constellation", constellation)?;
    header_dict.set_item("sampling_interval", sampling_interval)?;
    header_dict.set_item("leap_seconds", leap_seconds)?;
    header_dict.set_item("station", marker_name)?;
    header_dict.set_item("marker_type", marker_type)?;
    header_dict.set_item("rx_x", x_m)?;
    header_dict.set_item("rx_y", y_m)?;
    header_dict.set_item("rx_z", z_m)?;

    // extract observation data
    let obs_btree = match obs_rnx.record {
        Record::ObsRecord(r) => Some(r),
        _ => None,
    }
    .unwrap();
    let (time, (prn, (code, value))): (Vec<_>, (Vec<_>, (Vec<_>, Vec<_>))) = obs_btree
        .into_par_iter()
        .flat_map_iter(|(key, obs)| {
            let epoch_ms = key.epoch.to_unix_milliseconds() as i64;
            obs.signals
                .into_iter()
                .map(move |s| (epoch_ms, (s.sv, (s.observable, s.value))))
        })
        .filter(|(_, (sv, (code, _)))| {
            const_filter.contains(&sv.constellation)
                && (include_doppler || !code.is_doppler_observable())
        })
        .map(|(t, (sv, (code, value)))| (t, (sv.to_string(), (code.to_string(), value))))
        .unzip();

    Ok((
        header_dict.into(),
        PyArrowType(Int64Array::from(time).into_data()),
        PyArrowType(LargeStringArray::from(prn).into_data()),
        PyArrowType(LargeStringArray::from(code).into_data()),
        PyArrowType(Float64Array::from(value).into_data()),
    ))
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

fn get_nav_pos_optional(
    nav_rnx: &Rinex,
    epochs: Vec<Option<Epoch>>,
    svs: Vec<Option<SV>>,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let sv_set: FxHashSet<SV> = svs.iter().flatten().copied().collect();
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
            if epoch.is_none() || sv.is_none() {
                xs.push(f64::NAN);
                ys.push(f64::NAN);
                zs.push(f64::NAN);
                return;
            }
            let epoch = epoch.unwrap();
            let sv = sv.unwrap();
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

#[pyfunction]
fn _get_nav_coords(
    nav_fn: Vec<String>,
    time: PyArrowType<ArrayData>,
    prn: PyArrowType<ArrayData>,
) -> PyResult<(
    PyArrowType<ArrayData>,
    PyArrowType<ArrayData>,
    PyArrowType<ArrayData>,
)> {
    let nav_rnx = read_rinex_files(nav_fn)?;

    let time_array = time.0;
    let time_array = make_array(time_array);
    let time_array: &Int64Array = time_array
        .as_any()
        .downcast_ref()
        .ok_or_else(|| PyValueError::new_err("Time array must be of type Int64Array"))?;
    let prn_array = prn.0;
    let prn_array = make_array(prn_array);
    let prn_array: &LargeStringArray = prn_array
        .as_any()
        .downcast_ref()
        .ok_or_else(|| PyValueError::new_err("PRN array must be of type LargeStringArray"))?;

    let epochs: Vec<Option<Epoch>> = time_array
        .iter()
        .map(|t_ms| t_ms.and_then(|t| Some(Epoch::from_unix_milliseconds(t as f64))))
        .collect();
    let svs: Vec<Option<SV>> = prn_array
        .iter()
        .map(|prn_str| {
            prn_str.and_then(|s| match SV::from_str(s) {
                Ok(sv) => Some(sv),
                Err(_) => None,
            })
        })
        .collect();

    let (xs, ys, zs) = get_nav_pos_optional(&nav_rnx, epochs, svs);
    Ok((
        PyArrowType(Float64Array::from(xs).into_data()),
        PyArrowType(Float64Array::from(ys).into_data()),
        PyArrowType(Float64Array::from(zs).into_data()),
    ))
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_read_obs, m)?)?;
    m.add_function(wrap_pyfunction!(_get_nav_coords, m)?)?;
    Ok(())
}
