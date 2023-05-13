use pyo3::prelude::*;
use pyo3::{exceptions::PyValueError, AsPyPointer};

use tch;

fn wrap_tch_err(err: tch::TchError) -> PyErr {
    PyErr::new::<PyValueError, _>(format!("{err:?}"))
}

#[pyfunction]
fn add_one(t: PyObject) -> PyResult<PyObject> {
    let tensor = unsafe { tch::Tensor::pyobject_unpack(t.as_ptr() as *mut tch::python::CPyObject) };
    let tensor = tensor.map_err(wrap_tch_err)?;
    let tensor = match tensor {
        Some(tensor) => tensor,
        None => Err(PyErr::new::<PyValueError, _>("t is not a PyTorch tensor object"))?,
    };
    let tensor = tensor + 1.0;
    let tensor_ptr = tensor.pyobject_wrap().map_err(wrap_tch_err)?;
    let pyobject = Python::with_gil(|py| unsafe {
        PyObject::from_owned_ptr(py, tensor_ptr as *mut pyo3::ffi::PyObject)
    });
    Ok(pyobject)
}

/// A Python module implemented in Rust using tch to manipulate PyTorch
/// objects.
#[pymodule]
fn tch_ext(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(add_one, m)?)?;
    Ok(())
}
