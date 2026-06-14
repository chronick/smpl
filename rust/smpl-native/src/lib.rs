//! Optional native DSP for smpl, exposed via pyo3.
//!
//! Phase 0a ships a trivial-but-real function to prove the maturin/pyo3 build + import
//! path. The intent is that hot loops in `sample-analysis-core` / `smpl-audio` get ported
//! here behind the same interface ONLY where Python profiling shows a genuine hot spot —
//! the polars / pydantic-core idiom, not a rewrite.

use pyo3::prelude::*;

/// Root-mean-square level of an interleaved float sample buffer.
#[pyfunction]
fn rms(samples: Vec<f32>) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f64 = samples.iter().map(|&x| (x as f64) * (x as f64)).sum();
    (sum_sq / samples.len() as f64).sqrt()
}

/// Peak absolute amplitude of a float sample buffer.
#[pyfunction]
fn peak(samples: Vec<f32>) -> f32 {
    samples.iter().fold(0.0_f32, |acc, &x| acc.max(x.abs()))
}

/// Build identity — handy for the memo key's `op_version` of native ops.
#[pyfunction]
fn version() -> String {
    format!("smpl-native@{}", env!("CARGO_PKG_VERSION"))
}

#[pymodule]
fn smpl_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rms, m)?)?;
    m.add_function(wrap_pyfunction!(peak, m)?)?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    Ok(())
}
