use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

pub use tch;
pub use torch_sys;

pub struct PyTensor(pub tch::Tensor);

impl std::ops::Deref for PyTensor {
    type Target = tch::Tensor;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub fn wrap_tch_err(err: tch::TchError) -> PyErr {
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
        // There is no fallible alternative to ToPyObject/IntoPy at the moment, so we return
        // None on errors. https://github.com/PyO3/pyo3/issues/1813
        self.0.pyobject_wrap().map_or_else(
            |_| py.None(),
            |ptr| unsafe { PyObject::from_owned_ptr(py, ptr as *mut pyo3::ffi::PyObject) },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::types::IntoPyDict;

    #[test]
    fn rust_to_python() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let module = py.import("torch").unwrap();
            let tensor = tch::Tensor::from_slice(&[3, 1, 4, 1, 5]);
            let py_tensor = PyTensor(tensor);
            let py_obj = py_tensor.into_py(py).into_ref(py);
            assert_eq!(py_obj.get_type().name().unwrap(), "Tensor");
            assert!(py
                .eval(
                    "torch.is_tensor(tensor)",
                    None,
                    Some([("tensor", py_obj), ("torch", module)].into_py_dict(py))
                )
                .unwrap()
                .extract::<bool>()
                .unwrap());
        });
    }

    #[test]
    fn python_to_rust() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let module = py.import("torch").unwrap();
            let py_obj = py
                .eval(
                    "torch.tensor([3, 1, 4, 1, 5])",
                    None,
                    Some([("torch", module)].into_py_dict(py)),
                )
                .unwrap();
            let py_tensor = PyTensor::extract(py_obj).unwrap();
            let tensor = py_tensor.0;
            assert_eq!(tensor, tch::Tensor::from_slice(&[3, 1, 4, 1, 5]));
        });
    }
}
