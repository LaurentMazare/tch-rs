use pyo3::prelude::*;
use pyo3_tch::{wrap_tch_err, PyTensor};

#[pyfunction]
fn add_one(tensor: PyTensor) -> PyResult<PyTensor> {
    let tensor = tensor.f_add_scalar(1.0).map_err(wrap_tch_err)?;
    Ok(PyTensor(tensor))
}

/// A Python module implemented in Rust using tch to manipulate PyTorch
/// objects.
#[pymodule]
fn tch_ext(py: Python<'_>, m: &PyModule) -> PyResult<()> {
    py.import("torch")?;
    m.add_function(wrap_pyfunction!(add_one, m)?)?;
    Ok(())
}
