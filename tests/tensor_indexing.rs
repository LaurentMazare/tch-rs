use tch::{Tensor, Kind, Device};
use tch::{NewAxis, IndexOp};

#[test]
fn integer_index() {
    let opt = (Kind::Float, Device::Cpu);

    let tensor = Tensor::arange1(0, 2 * 3, opt)
        .view(&[2, 3]);
    let result = tensor.i(1);
    let expect = Tensor::arange1(3, 6, opt);
    assert!(result.eq1(&expect).all().int64_value(&[]) == 1);

    let tensor = Tensor::arange1(0, 2 * 3, opt)
        .view(&[2, 3]);
    let result = tensor.i((.., 2));
    let expect = Tensor::of_slice(&[2_f32, 5.])
        .view(&[2]);
    assert!(result.eq1(&expect).all().int64_value(&[]) == 1);
}

#[test]
fn range_index() {
    let opt = (Kind::Float, Device::Cpu);

    // Range
    let tensor = Tensor::arange1(0, 4 * 3, opt)
        .view(&[4, 3]);
    let result = tensor.i(1..3);
    let expect = Tensor::arange1(1 * 3, 3 * 3, opt)
        .view(&[2, 3]);
    assert!(result.eq1(&expect).all().int64_value(&[]) == 1);

    // RangeFull
    let tensor = Tensor::arange1(0, 4 * 3, opt)
        .view(&[4, 3]);
    let result = tensor.i(..);
    let expect = Tensor::arange1(0, 4 * 3, opt)
        .view(&[4, 3]);
    assert!(result.eq1(&expect).all().int64_value(&[]) == 1);

    // RangeFrom
    let tensor = Tensor::arange1(0, 4 * 3, opt)
        .view(&[4, 3]);
    let result = tensor.i(2..);
    let expect = Tensor::arange1(2 * 3, 4 * 3, opt)
        .view(&[2, 3]);
    assert!(result.eq1(&expect).all().int64_value(&[]) == 1);

    // RangeTo
    let tensor = Tensor::arange1(0, 4 * 3, opt)
        .view(&[4, 3]);
    let result = tensor.i(..2);
    let expect = Tensor::arange1(0, 2 * 3, opt)
        .view(&[2, 3]);
    assert!(result.eq1(&expect).all().int64_value(&[]) == 1);

    // RangeInclusive
    let tensor = Tensor::arange1(0, 4 * 3, opt)
        .view(&[4, 3]);
    let result = tensor.i(1..=2);
    let expect = Tensor::arange1(1 * 3, 3 * 3, opt)
        .view(&[2, 3]);
    assert!(result.eq1(&expect).all().int64_value(&[]) == 1);

    // RangeTo
    let tensor = Tensor::arange1(0, 4 * 3, opt)
        .view(&[4, 3]);
    let result = tensor.i(..1);
    let expect = Tensor::arange1(0, 1 * 3, opt)
        .view(&[1, 3]);
    assert!(result.eq1(&expect).all().int64_value(&[]) == 1);

    // RangeToInclusive
    let tensor = Tensor::arange1(0, 4 * 3, opt)
        .view(&[4, 3]);
    let result = tensor.i(..=1);
    let expect = Tensor::arange1(0, 2 * 3, opt)
        .view(&[2, 3]);
    assert!(result.eq1(&expect).all().int64_value(&[]) == 1);
}

#[test]
fn slice_index() {
    let opt = (Kind::Float, Device::Cpu);

    let tensor = Tensor::arange1(0, 6 * 2, opt)
        .view(&[6, 2]);
    let index: &[_] = &[1, 3, 5];
    let result = tensor.i(index);
    let expect = Tensor::of_slice(&[2_f32, 3., 6., 7., 10., 11.])
        .view(&[3, 2]);
    assert!(result.eq1(&expect).all().int64_value(&[]) == 1);

    let tensor = Tensor::arange1(0, 3 * 4, opt)
        .view(&[3, 4]);
    let index: &[_] = &[3, 0];
    let result = tensor.i((.., index));
    let expect = Tensor::of_slice(&[3_f32, 0., 7., 4., 11., 8.])
        .view(&[3, 2]);
    assert!(result.eq1(&expect).all().int64_value(&[]) == 1);
}

#[test]
fn new_index() {
    let opt = (Kind::Float, Device::Cpu);

    let tensor = Tensor::arange1(0, 2 * 3, opt)
        .view(&[2, 3]);
    let result = tensor.i((NewAxis,));
    let expect = Tensor::arange1(0, 1 * 2 * 3, opt)
        .view(&[1, 2, 3]);
    assert!(result.eq1(&expect).all().int64_value(&[]) == 1);

    let tensor = Tensor::arange1(0, 2 * 3, opt)
        .view(&[2, 3]);
    let result = tensor.i((.., NewAxis));
    let expect = Tensor::arange1(0, 2 * 1 * 3, opt)
        .view(&[2, 1, 3]);
    assert!(result.eq1(&expect).all().int64_value(&[]) == 1);

    let tensor = Tensor::arange1(0, 2 * 3, opt)
        .view(&[2, 3]);
    let result = tensor.i((.., .., NewAxis,));
    let expect = Tensor::arange1(0, 2 * 3 * 1, opt)
        .view(&[2, 3, 1]);
    assert!(result.eq1(&expect).all().int64_value(&[]) == 1);
}

#[test]
fn complex_index() {
    let opt = (Kind::Float, Device::Cpu);

    let tensor = Tensor::arange1(0, 2 * 3 * 5 * 7, opt)
        .view(&[2, 3, 5, 7]);
    let result = tensor.i((1, 1..2, vec![2, 3, 0].as_slice(), NewAxis, 3..));
    let expect = Tensor::of_slice(&[157_f32, 158., 159., 160., 164., 165., 166., 167., 143., 144., 145., 146.])
        .view(&[1, 3, 1, 4]);
    assert!(result.eq1(&expect).all().int64_value(&[]) == 1);
}
