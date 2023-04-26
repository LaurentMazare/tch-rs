use tch::Tensor;

pub fn from<'a, T>(t: &'a Tensor) -> T
where
    <T as TryFrom<&'a tch::Tensor>>::Error: std::fmt::Debug,
    T: TryFrom<&'a Tensor>,
{
    T::try_from(t).unwrap()
}

#[allow(dead_code)]
pub fn f64_from(t: &Tensor) -> f64 {
    from::<f64>(t)
}
