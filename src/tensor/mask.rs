use crate::Tensor;

pub trait MaskSelectOp {
    fn m<M>(&self, mask: M) -> Self
    where
        M: Into<MaskSelector>;
}

impl MaskSelectOp for Tensor {
    fn m<M>(&self, mask: M) -> Self
    where
        M: Into<MaskSelector>,
    {
        let mask = mask.into();
        self.masked_select(&mask.0)
    }
}

#[derive(Debug)]
pub struct MaskSelector(Tensor);

impl From<Tensor> for MaskSelector {
    fn from(value: Tensor) -> Self {
        Self(value)
    }
}

impl From<&Tensor> for MaskSelector {
    fn from(value: &Tensor) -> Self {
        Self(value.shallow_clone())
    }
}

impl From<Vec<bool>> for MaskSelector {
    fn from(value: Vec<bool>) -> Self {
        Self(Tensor::try_from(value).unwrap())
    }
}

impl From<&Vec<bool>> for MaskSelector {
    fn from(value: &Vec<bool>) -> Self {
        Self(Tensor::try_from(value).unwrap())
    }
}

impl From<&[bool]> for MaskSelector {
    fn from(value: &[bool]) -> Self {
        Self(Tensor::try_from(value).unwrap())
    }
}

impl<const N: usize> From<[bool; N]> for MaskSelector {
    fn from(value: [bool; N]) -> Self {
        Self(Tensor::try_from(value).unwrap())
    }
}

impl<const N: usize> From<&[bool; N]> for MaskSelector {
    fn from(value: &[bool; N]) -> Self {
        Self(Tensor::try_from(value).unwrap())
    }
}

impl<const N1: usize, const N2: usize> From<[[bool; N2]; N1]> for MaskSelector {
    fn from(value: [[bool; N2]; N1]) -> Self {
        Self(Tensor::try_from(value).unwrap())
    }
}

impl<const N1: usize, const N2: usize, const N3: usize> From<[[[bool; N3]; N2]; N1]>
    for MaskSelector
{
    fn from(value: [[[bool; N3]; N2]; N1]) -> Self {
        Self(Tensor::try_from(value).unwrap())
    }
}

impl<const N1: usize, const N2: usize, const N3: usize, const N4: usize>
    From<[[[[bool; N4]; N3]; N2]; N1]> for MaskSelector
{
    fn from(value: [[[[bool; N4]; N3]; N2]; N1]) -> Self {
        Self(Tensor::try_from(value).unwrap())
    }
}
