use cpython::{NoArgs, ObjectProtocol, PyObject, PyResult, Python};
use tch::Tensor;

#[derive(Debug)]
pub struct Step {
    pub obs: Tensor,
    pub action: i64,
    pub reward: f64,
    pub is_done: bool,
}

impl Step {
    pub fn copy_with_obs(&self, obs: &Tensor) -> Step {
        Step {
            obs: obs.copy(),
            action: self.action,
            reward: self.reward,
            is_done: self.is_done,
        }
    }
}

pub struct GymEnv {
    env: PyObject,
}

impl GymEnv {
    pub fn new() -> PyResult<GymEnv> {
        let gil = Python::acquire_gil();
        let py = gil.python();
        let gym = py.import("gym")?;
        let env = gym.call(py, "make", ("CartPole-v0",), None)?;
        let _ = env.call_method(py, "seed", (42,), None)?;
        Ok(GymEnv { env })
    }

    pub fn reset(&self) -> PyResult<Tensor> {
        let gil = Python::acquire_gil();
        let py = gil.python();
        let obs = self.env.call_method(py, "reset", NoArgs, None)?;
        Ok(Tensor::float_vec(&obs.extract::<Vec<f64>>(py)?))
    }

    pub fn step(&self, action: i64) -> PyResult<Step> {
        let gil = Python::acquire_gil();
        let py = gil.python();
        let step = self.env.call_method(py, "step", (action,), None)?;
        Ok(Step {
            obs: Tensor::float_vec(&step.get_item(py, 0)?.extract::<Vec<f64>>(py)?),
            reward: step.get_item(py, 1)?.extract(py)?,
            is_done: step.get_item(py, 2)?.extract(py)?,
            action,
        })
    }
}
