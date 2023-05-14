use pyo3::prelude::*;
use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    AsPyPointer,
};

struct PyTensor(tch::Tensor);

fn wrap_tch_err(err: tch::TchError) -> PyErr {
    PyErr::new::<PyValueError, _>(format!("{err:?}"))
}

impl<'source> FromPyObject<'source> for PyTensor {
    fn extract(ob: &'source PyAny) -> PyResult<Self> {
        let ptr = ob.as_ptr() as *mut tch::python::CPyObject;
        let tensor = unsafe { tch::Tensor::pyobject_unpack(ptr) };
        tensor
            .map_err(wrap_tch_err)?
            .ok_or_else(|| {
                let type_ = ob.get_type();
                PyErr::new::<PyTypeError, _>(format!("expected a torch.Tensor, got {type_}"))
            })
            .map(PyTensor)
    }
}

impl IntoPy<PyObject> for PyTensor {
    fn into_py(self, py: Python<'_>) -> PyObject {
        // There is no fallible alternative to ToPyObject/IntoPy at the moment so we return
        // None on errors. https://github.com/PyO3/pyo3/issues/1813
        self.0.pyobject_wrap().map_or_else(
            |_| py.None(),
            |ptr| unsafe { PyObject::from_owned_ptr(py, ptr as *mut pyo3::ffi::PyObject) },
        )
    }
}

#[pyfunction]
fn add_one(tensor: PyTensor) -> PyResult<PyTensor> {
    let tensor = tensor.0.f_add_scalar(1.0).map_err(wrap_tch_err)?;
    Ok(PyTensor(tensor))
}

/// A Python module implemented in Rust using tch to manipulate PyTorch
/// objects.
#[pymodule]
fn tch_ext(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(add_one, m)?)?;
    Ok(())
}
