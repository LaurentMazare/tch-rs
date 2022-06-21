//! Layers defined by closures.

/// A layer defined by a simple closure.
pub struct Func<'a, Input, Output> {
    f: Box<dyn 'a + Fn(&Input) -> Output + Send>,
}

impl<'a, Input, Output> std::fmt::Debug for Func<'a, Input, Output> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "func")
    }
}

pub fn func<'a, F, Input, Output>(f: F) -> Func<'a, Input, Output>
where
    F: 'a + Fn(&Input) -> Output + Send,
{
    Func { f: Box::new(f) }
}

impl<'a, Input, Output> super::module::Module for Func<'a, Input, Output> {
    type Input = Input;
    type Output = Output;

    fn forward(&self, xs: &Self::Input) -> Self::Output {
        (*self.f)(xs)
    }
}

/// A layer defined by a closure with an additional training parameter.
pub struct FuncT<'a, Input, Output> {
    f: Box<dyn 'a + Fn(&Input, bool) -> Output + Send>,
    batch_accuracy_f: Box<dyn 'a + Fn(&Self, &Input, &Input, crate::Device, i64) -> f64 + Send>,
}

impl<'a, Input, Output> std::fmt::Debug for FuncT<'a, Input, Output> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "funcT")
    }
}

pub fn func_t<'a, F, F2, Input, Output>(f: F, batch_accuracy_f: F2) -> FuncT<'a, Input, Output>
where
    F: 'a + Fn(&Input, bool) -> Output + Send,
    F2: 'a + Fn(&FuncT<Input, Output>, &Input, &Input, crate::Device, i64) -> f64 + Send,
{
    FuncT { f: Box::new(f), batch_accuracy_f: Box::new(batch_accuracy_f) }
}

impl<'a, Input, Output> super::module::ModuleT for FuncT<'a, Input, Output> {
    type Input = Input;
    type Output = Output;

    fn forward_t(&self, xs: &Self::Input, train: bool) -> Self::Output {
        (*self.f)(xs, train)
    }

    fn batch_accuracy_for_logits(
        &self,
        xs: &Self::Input,
        ys: &Self::Input,
        d: crate::Device,
        batch_size: i64,
    ) -> f64 {
        (*self.batch_accuracy_f)(self, xs, ys, d, batch_size)
    }
}
