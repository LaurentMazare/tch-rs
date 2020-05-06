use super::Tensor;
use crate::TchError;

pub struct Iter<T> {
    index: i64,
    len: i64,
    content: Tensor,
    phantom: std::marker::PhantomData<T>,
}

impl Tensor {
    pub fn iter<T>(&self) -> Result<Iter<T>, TchError> {
        Ok(Iter {
            index: 0,
            len: self.size1()?,
            content: self.shallow_clone(),
            phantom: std::marker::PhantomData,
        })
    }
}

impl std::iter::Iterator for Iter<i64> {
    type Item = i64;
    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.len {
            return None;
        }
        let v = self.content.int64_value(&[self.index]);
        self.index += 1;
        Some(v)
    }
}

impl std::iter::Iterator for Iter<f64> {
    type Item = f64;
    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.len {
            return None;
        }
        let v = self.content.double_value(&[self.index]);
        self.index += 1;
        Some(v)
    }
}
