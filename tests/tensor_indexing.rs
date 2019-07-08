use tch::{Device, Kind, Tensor};
use tch::{IndexOp, NewAxis};

#[test]
fn integer_index() {
    let opt = (Kind::Float, Device::Cpu);

    let tensor = Tensor::arange1(0, 2 * 3, opt).view(&[2, 3]);
    let result = tensor.i(1);
    assert_eq!(result.size(), &[3]);
    assert_eq!(Vec::<i64>::from(result), &[3, 4, 5]);

    let tensor = Tensor::arange1(0, 2 * 3, opt).view(&[2, 3]);
    let result = tensor.i((.., 2));
    assert_eq!(result.size(), &[2]);
    assert_eq!(Vec::<i64>::from(result), &[2, 5]);
}

#[test]
fn range_index() {
    let opt = (Kind::Float, Device::Cpu);

    // Range
    let tensor = Tensor::arange1(0, 4 * 3, opt).view(&[4, 3]);
    let result = tensor.i(1..3);
    assert_eq!(result.size(), &[2, 3]);
    assert_eq!(Vec::<i64>::from(result), &[3, 4, 5, 6, 7, 8]);

    // RangeFull
    let tensor = Tensor::arange1(0, 2 * 3, opt).view(&[2, 3]);
    let result = tensor.i(..);
    assert_eq!(result.size(), &[2, 3]);
    assert_eq!(Vec::<i64>::from(result), &[0, 1, 2, 3, 4, 5]);

    // RangeFrom
    let tensor = Tensor::arange1(0, 4 * 3, opt).view(&[4, 3]);
    let result = tensor.i(2..);
    assert_eq!(result.size(), &[2, 3]);
    assert_eq!(Vec::<i64>::from(result), &[6, 7, 8, 9, 10, 11]);

    // RangeTo
    let tensor = Tensor::arange1(0, 4 * 3, opt).view(&[4, 3]);
    let result = tensor.i(..2);
    assert_eq!(result.size(), &[2, 3]);
    assert_eq!(Vec::<i64>::from(result), &[0, 1, 2, 3, 4, 5]);

    // RangeInclusive
    let tensor = Tensor::arange1(0, 4 * 3, opt).view(&[4, 3]);
    let result = tensor.i(1..=2);
    assert_eq!(result.size(), &[2, 3]);
    assert_eq!(Vec::<i64>::from(result), &[3, 4, 5, 6, 7, 8]);

    // RangeTo
    let tensor = Tensor::arange1(0, 4 * 3, opt).view(&[4, 3]);
    let result = tensor.i(..1);
    assert_eq!(result.size(), &[1, 3]);
    assert_eq!(Vec::<i64>::from(result), &[0, 1, 2]);

    // RangeToInclusive
    let tensor = Tensor::arange1(0, 4 * 3, opt).view(&[4, 3]);
    let result = tensor.i(..=1);
    assert_eq!(result.size(), &[2, 3]);
    assert_eq!(Vec::<i64>::from(result), &[0, 1, 2, 3, 4, 5]);
}

#[test]
fn slice_index() {
    let opt = (Kind::Float, Device::Cpu);

    let tensor = Tensor::arange1(0, 6 * 2, opt).view(&[6, 2]);
    let index: &[_] = &[1, 3, 5];
    let result = tensor.i(index);
    assert_eq!(result.size(), &[3, 2]);
    assert_eq!(Vec::<i64>::from(result), &[2, 3, 6, 7, 10, 11]);

    let tensor = Tensor::arange1(0, 3 * 4, opt).view(&[3, 4]);
    let index: &[_] = &[3, 0];
    let result = tensor.i((.., index));
    assert_eq!(result.size(), &[3, 2]);
    assert_eq!(Vec::<i64>::from(result), &[3, 0, 7, 4, 11, 8]);
}

#[test]
fn new_index() {
    let opt = (Kind::Float, Device::Cpu);

    let tensor = Tensor::arange1(0, 2 * 3, opt).view(&[2, 3]);
    let result = tensor.i((NewAxis,));
    assert_eq!(result.size(), &[1, 2, 3]);
    assert_eq!(Vec::<i64>::from(result), &[0, 1, 2, 3, 4, 5]);

    let tensor = Tensor::arange1(0, 2 * 3, opt).view(&[2, 3]);
    let result = tensor.i((.., NewAxis));
    assert_eq!(result.size(), &[2, 1, 3]);
    assert_eq!(Vec::<i64>::from(result), &[0, 1, 2, 3, 4, 5]);

    let tensor = Tensor::arange1(0, 2 * 3, opt).view(&[2, 3]);
    let result = tensor.i((.., .., NewAxis));
    assert_eq!(result.size(), &[2, 3, 1]);
    assert_eq!(Vec::<i64>::from(result), &[0, 1, 2, 3, 4, 5]);
}

#[test]
fn complex_index() {
    let opt = (Kind::Float, Device::Cpu);

    let tensor = Tensor::arange1(0, 2 * 3 * 5 * 7, opt).view(&[2, 3, 5, 7]);
    let result = tensor.i((1, 1..2, vec![2, 3, 0].as_slice(), NewAxis, 3..));
    assert_eq!(result.size(), &[1, 3, 1, 4]);
    assert_eq!(
        Vec::<i64>::from(result),
        &[157, 158, 159, 160, 164, 165, 166, 167, 143, 144, 145, 146]
    );
}
