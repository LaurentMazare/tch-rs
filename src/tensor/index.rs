use std::ops::{
    RangeBounds,
    Range,
    RangeFrom,
    RangeFull,
    RangeInclusive,
    RangeTo,
    RangeToInclusive,
    Bound,
};
use super::Tensor;
use failure::Fallible;

pub enum TensorIndexer<'a> {
    Select(i64),
    IndexSelect(&'a [i64]),
    MaskedSelect(&'a [bool]),
    TensorSelect(&'a Tensor),
    Narrow(Bound<i64>, Bound<i64>),
}

impl<'a> From<i64> for TensorIndexer<'a> {
    fn from(index: i64) -> Self {
        TensorIndexer::Select(index)
    }
}

impl<'a> From<&'a [i64]> for TensorIndexer<'a> {
    fn from(indexes: &'a [i64]) -> Self {
        TensorIndexer::IndexSelect(indexes)
    }
}

macro_rules! impl_from_range {
    ($range_type:ty) => {
        impl<'a> From<$range_type> for TensorIndexer<'a> {
            fn from(range: $range_type) -> Self {
                use std::ops::Bound::*;

                let start = match range.start_bound() {
                    Included(idx) => Included(*idx),
                    Excluded(idx) => Excluded(*idx),
                    Unbounded => Unbounded,
                };

                let end = match range.end_bound() {
                    Included(idx) => Included(*idx),
                    Excluded(idx) => Excluded(*idx),
                    Unbounded => Unbounded,
                };

                TensorIndexer::Narrow(start, end)
            }
        }
    }
}

impl_from_range!(Range<i64>);
impl_from_range!(RangeFrom<i64>);
impl_from_range!(RangeFull);
impl_from_range!(RangeInclusive<i64>);
impl_from_range!(RangeTo<i64>);
impl_from_range!(RangeToInclusive<i64>);

impl Tensor {
    fn indexer<'a>(&self, index_spec: &[TensorIndexer<'a>]) -> Tensor {
        use std::ops::Bound::*;
        use TensorIndexer::*;

        assert!(
            index_spec.len() <= self.size().len(),
            format!("too many indices for tensor of dimension {}", self.size().len())
        );

        let mut result_tensor = self.shallow_clone();
        let mut result_dim: i64 = 0;

        for (index_dim, spec) in index_spec.iter().enumerate() {
            let dim_len = result_tensor.size()[result_dim as usize] as i64;

            let (next_tensor, next_dim) = match spec {
                Select(index) => {
                    (result_tensor.select(result_dim, *index), result_dim)
                }
                IndexSelect(indexes) => {
                    let index_tensor = Tensor::of_slice(indexes);
                    (
                        result_tensor.index_select(result_dim, &index_tensor),
                        result_dim + 1,
                    )
                }
                MaskedSelect(mask) => {
                    assert!(
                        mask.len() as i64 == dim_len,
                        format!("The length of the mask [{}] does not match the shape of the indexed tensor {:?} at index {}", mask.len(), self.size(), index_dim),
                    );

                    let mut indexes = vec![];
                    for (idx, selected) in mask.iter().enumerate() {
                        if *selected {
                            indexes.push(idx as i64);
                        }
                    }
                    let index_tensor = Tensor::of_slice(&indexes);
                    (
                        result_tensor.index_select(result_dim, &index_tensor),
                        result_dim + 1,
                    )
                }
                TensorSelect(tensor) => {
                    use super::Kind::*;

                    match tensor.kind() {
                        Int64 => { // index select
                            // TODO
                        }
                        Uint8 => { // masked select
                            // TODO
                        }
                        _ => {
                            panic!("the kind of tensors used as indices must be {:?} or {:?}", Int64, Uint8);
                        }
                    }
                }
                Narrow(Unbounded, Unbounded) => (
                    result_tensor,
                    result_dim + 1,
                ),
                Narrow(Included(start), Unbounded) => (
                    result_tensor.narrow(result_dim, *start, dim_len - *start),
                    result_dim + 1,
                ),
                Narrow(Excluded(start), Unbounded) => (
                    result_tensor.narrow(result_dim, *start + 1, dim_len - *start - 1),
                    result_dim + 1,
                ),
                Narrow(Unbounded, Included(end)) => (
                    result_tensor.narrow(result_dim, 0, *end + 1),
                    result_dim + 1,
                ),
                Narrow(Unbounded, Excluded(end)) => (
                    result_tensor.narrow(result_dim, 0, *end),
                    result_dim + 1,
                ),
                Narrow(Included(start), Included(end)) => (
                    result_tensor.narrow(result_dim, *start, *end - *start + 1),
                    result_dim + 1,
                ),
                Narrow(Included(start), Excluded(end)) => (
                    result_tensor.narrow(result_dim, *start, *end - *start),
                    result_dim + 1,
                ),
                Narrow(Excluded(start), Included(end)) => (
                    result_tensor.narrow(result_dim, *start + 1, *end - *start),
                    result_dim + 1,
                ),
                Narrow(Excluded(start), Excluded(end)) => (
                    result_tensor.narrow(result_dim, *start + 1, *end - *start - 1),
                    result_dim + 1,
                ),
            };

            result_tensor = next_tensor;
            result_dim = next_dim;
        }

        result_tensor
    }
}

trait IndexOp<T> {
    fn i(&self, index: T) -> Tensor;
}


impl<'a, A> IndexOp<A> for Tensor where
    A: Into<TensorIndexer<'a>>,
{
    fn i(&self, index: A) -> Tensor {
        self.indexer(&[index.into()])
    }
}

impl<'a, A> IndexOp<(A,)> for Tensor where
    A: Into<TensorIndexer<'a>>,
{
    fn i(&self, index: (A,)) -> Tensor {
        let a = index.0.into();
        self.indexer(&[a])
    }
}

impl<'a, 'b, A, B> IndexOp<(A, B)> for Tensor where
        A: Into<TensorIndexer<'a>>,
        B: Into<TensorIndexer<'b>>,
{
    fn i(&self, index: (A, B)) -> Tensor {
        let a = index.0.into();
        let b = index.1.into();
        self.indexer(&[a, b])
    }
}

impl<'a, 'b, 'c, A, B, C> IndexOp<(A, B, C)> for Tensor where
        A: Into<TensorIndexer<'a>>,
        B: Into<TensorIndexer<'b>>,
        C: Into<TensorIndexer<'c>>,
{
    fn i(&self, index: (A, B, C)) -> Tensor {
        let a = index.0.into();
        let b = index.1.into();
        let c = index.2.into();
        self.indexer(&[a, b, c])
    }
}

impl<'a, 'b, 'c, 'd, A, B, C, D> IndexOp<(A, B, C, D)> for Tensor where
        A: Into<TensorIndexer<'a>>,
        B: Into<TensorIndexer<'b>>,
        C: Into<TensorIndexer<'c>>,
        D: Into<TensorIndexer<'d>>,
{
    fn i(&self, index: (A, B, C, D)) -> Tensor {
        let a = index.0.into();
        let b = index.1.into();
        let c = index.2.into();
        let d = index.3.into();
        self.indexer(&[a, b, c, d])
    }
}

impl<'a, 'b, 'c, 'd, 'e, A, B, C, D, E> IndexOp<(A, B, C, D, E)> for Tensor where
        A: Into<TensorIndexer<'a>>,
        B: Into<TensorIndexer<'b>>,
        C: Into<TensorIndexer<'c>>,
        D: Into<TensorIndexer<'d>>,
        E: Into<TensorIndexer<'e>>,
{
    fn i(&self, index: (A, B, C, D, E)) -> Tensor {
        let a = index.0.into();
        let b = index.1.into();
        let c = index.2.into();
        let d = index.3.into();
        let e = index.4.into();
        self.indexer(&[a, b, c, d, e])
    }
}

impl<'a, 'b, 'c, 'd, 'e, 'f, A, B, C, D, E, F> IndexOp<(A, B, C, D, E, F)> for Tensor where
        A: Into<TensorIndexer<'a>>,
        B: Into<TensorIndexer<'b>>,
        C: Into<TensorIndexer<'c>>,
        D: Into<TensorIndexer<'d>>,
        E: Into<TensorIndexer<'e>>,
        F: Into<TensorIndexer<'f>>,
{
    fn i(&self, index: (A, B, C, D, E, F)) -> Tensor {
        let a = index.0.into();
        let b = index.1.into();
        let c = index.2.into();
        let d = index.3.into();
        let e = index.4.into();
        let f = index.5.into();
        self.indexer(&[a, b, c, d, e, f])
    }
}
