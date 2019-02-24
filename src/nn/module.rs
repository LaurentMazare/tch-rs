// TODO: add layer names when registering inner modules?
use crate::tensor::Tensor;
use crate::Device;

pub trait Module {
    fn forward(&self, xs: &Tensor) -> Tensor;
}

pub trait ModuleT {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor;

    fn batch_accuracy_for_logits(
        &self,
        xs: &Tensor,
        ys: &Tensor,
        d: Device,
        batch_size: i64,
    ) -> f64 {
        let xs_size = xs.size()[0] as i64;
        let mut sum_accuracy = 0f64;
        let mut sample_count = 0f64;
        for index in 0..(xs_size / batch_size) {
            let start = index * batch_size;
            let xs = xs.narrow(0, start, batch_size);
            let ys = ys.narrow(0, start, batch_size);
            let acc = self
                .forward_t(&xs.to_device(d), false)
                .accuracy_for_logits(&ys.to_device(d));
            let cnt = xs.size()[0] as f64;
            sum_accuracy += f64::from(&acc) * cnt;
            sample_count += cnt;
        }
        sum_accuracy / sample_count
    }
}

impl<T> ModuleT for T
where
    T: Module,
{
    fn forward_t(&self, xs: &Tensor, _train: bool) -> Tensor {
        self.forward(&xs)
    }
}

impl Tensor {
    pub fn apply<M: Module>(&self, m: &M) -> Tensor {
        m.forward(&self)
    }
    pub fn apply_t<M: ModuleT>(&self, m: &M, train: bool) -> Tensor {
        m.forward_t(&self, train)
    }
}
