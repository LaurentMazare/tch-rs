use std::ops::{Div, Mul};
use pyo3::prelude::*;
use tch::{Device, Kind, Scalar, Tensor};

#[pyclass(name = "EntropyMetric")]
pub struct EntropyMetric {
    pub counter: Tensor,
}

#[pymethods]
impl EntropyMetric {
    #[new]
    fn __new__(n_classes: i64) -> Self {
        EntropyMetric {
            counter: Tensor::zeros(&[n_classes], (Kind::Float, Device::Cpu))
        }
    }

    fn update(&mut self, x: Tensor) -> PyResult<()> {
        let ones = Tensor::ones(&[x.size()[0]], (Kind::Float, Device::Cpu));
        self.counter = self.counter.scatter_add(0, &x, &ones);

        Ok(())
    }

    fn get_counter(&self) -> PyResult<&Tensor> {
        Ok(&self.counter)
    }

    fn compute(&self) -> PyResult<Tensor> {
        let counts = self.counter.masked_select(&self.counter.gt(Scalar::from(0.0)));
        let probs = counts.div(&self.counter.sum(Kind::Float));
        let entropy = (probs.neg().mul(probs.log())).sum(Kind::Float);
        Ok(entropy)
    }
}

#[pymodule]
fn python_entropy(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<EntropyMetric>()?;
    Ok(())
}