use anyhow::Result;
use std::convert::TryFrom;
use tch::{Device, Kind, Tensor};
use tch::{IndexOp, NewAxis};

#[test]
fn integer_index() -> Result<()> {
    let opt = (Kind::Int64, Device::Cpu);

    let tensor = Tensor::arange1(0, 2 * 3, opt).view([2, 3]);
    let result = tensor.i(1);
    assert_eq!(result.size(), &[3]);
    assert_eq!(Vec::<i64>::try_from(result)?, &[3, 4, 5]);

    let tensor = Tensor::arange1(0, 2 * 3, opt).view([2, 3]);
    let result = tensor.i((.., 2));
    assert_eq!(result.size(), &[2]);
    assert_eq!(Vec::<i64>::try_from(result)?, &[2, 5]);

    let result = tensor.i((.., -2));
    assert_eq!(result.size(), &[2]);
    assert_eq!(Vec::<i64>::try_from(result)?, &[1, 4]);
    Ok(())
}

#[test]
fn range_index() -> Result<()> {
    let opt = (Kind::Int64, Device::Cpu);

    // Range
    let tensor = Tensor::arange1(0, 4 * 3, opt).view([4, 3]);
    let result = tensor.i(1..3);
    assert_eq!(result.size(), &[2, 3]);
    assert_eq!(
        Vec::<Vec<i64>>::try_from(result)?,
        &[vec![3, 4, 5], vec![6, 7, 8]]
    );

    // RangeFull
    let tensor = Tensor::arange1(0, 2 * 3, opt).view([2, 3]);
    let result = tensor.i(..);
    assert_eq!(result.size(), &[2, 3]);
    assert_eq!(
        Vec::<Vec<i64>>::try_from(result)?,
        &[vec![0, 1, 2], vec![3, 4, 5]]
    );

    // RangeFrom
    let tensor = Tensor::arange1(0, 4 * 3, opt).view([4, 3]);
    let result = tensor.i(2..);
    assert_eq!(result.size(), &[2, 3]);
    assert_eq!(
        Vec::<Vec<i64>>::try_from(result)?,
        &[vec![6, 7, 8], vec![9, 10, 11]]
    );

    // RangeTo
    let tensor = Tensor::arange1(0, 4 * 3, opt).view([4, 3]);
    let result = tensor.i(..2);
    assert_eq!(result.size(), &[2, 3]);
    assert_eq!(
        Vec::<Vec<i64>>::try_from(result)?,
        &[vec![0, 1, 2], vec![3, 4, 5]]
    );

    // RangeInclusive
    let tensor = Tensor::arange1(0, 4 * 3, opt).view([4, 3]);
    let result = tensor.i(1..=2);
    assert_eq!(result.size(), &[2, 3]);
    assert_eq!(
        Vec::<Vec<i64>>::try_from(result)?,
        &[vec![3, 4, 5], vec![6, 7, 8]]
    );

    // RangeTo
    let tensor = Tensor::arange1(0, 4 * 3, opt).view([4, 3]);
    let result = tensor.i(..1);
    assert_eq!(result.size(), &[1, 3]);
    assert_eq!(Vec::<Vec<i64>>::try_from(result)?, &[vec![0, 1, 2]]);

    // RangeToInclusive
    let tensor = Tensor::arange1(0, 4 * 3, opt).view([4, 3]);
    let result = tensor.i(..=1);
    assert_eq!(result.size(), &[2, 3]);
    assert_eq!(
        Vec::<Vec<i64>>::try_from(result)?,
        &[vec![0, 1, 2], vec![3, 4, 5]]
    );
    Ok(())
}

#[test]
fn slice_index() -> Result<()> {
    let opt = (Kind::Int64, Device::Cpu);

    let tensor = Tensor::arange1(0, 6 * 2, opt).view([6, 2]);
    let index: &[_] = &[1, 3, 5];
    let result = tensor.i(index);
    assert_eq!(result.size(), &[3, 2]);
    assert_eq!(
        Vec::<Vec<i64>>::try_from(result)?,
        &[vec![2, 3], vec![6, 7], vec![10, 11]]
    );

    let tensor = Tensor::arange1(0, 3 * 4, opt).view([3, 4]);
    let index: &[_] = &[3, 0];
    let result = tensor.i((.., index));
    assert_eq!(result.size(), &[3, 2]);
    assert_eq!(
        Vec::<Vec<i64>>::try_from(result)?,
        &[vec![3, 0], vec![7, 4], vec![11, 8]]
    );
    Ok(())
}

#[test]
fn new_index() -> Result<()> {
    let opt = (Kind::Int64, Device::Cpu);

    let tensor = Tensor::arange1(0, 2 * 3, opt).view([2, 3]);
    let result = tensor.i((NewAxis,));
    assert_eq!(result.size(), &[1, 2, 3]);
    assert_eq!(
        Vec::<Vec<Vec<i64>>>::try_from(result)?,
        &[vec![vec![0, 1, 2], vec![3, 4, 5]]]
    );

    let tensor = Tensor::arange1(0, 2 * 3, opt).view([2, 3]);
    let result = tensor.i((.., NewAxis));
    assert_eq!(result.size(), &[2, 1, 3]);
    assert_eq!(
        Vec::<Vec<Vec<i64>>>::try_from(result)?,
        &[vec![vec![0, 1, 2]], vec![vec![3, 4, 5]]]
    );

    let tensor = Tensor::arange1(0, 2 * 3, opt).view([2, 3]);
    let result = tensor.i((.., .., NewAxis));
    assert_eq!(result.size(), &[2, 3, 1]);
    assert_eq!(
        Vec::<Vec<Vec<i64>>>::try_from(result)?,
        &[
            vec![vec![0], vec![1], vec![2]],
            vec![vec![3], vec![4], vec![5]]
        ]
    );
    Ok(())
}

#[test]
fn complex_index() -> Result<()> {
    let opt = (Kind::Int64, Device::Cpu);

    let tensor = Tensor::arange1(0, 2 * 3 * 5 * 7, opt).view([2, 3, 5, 7]);
    let result = tensor.i((1, 1..2, vec![2, 3, 0].as_slice(), NewAxis, 3..));
    assert_eq!(result.size(), &[1, 3, 1, 4]);
    assert_eq!(
        Vec::<Vec<Vec<Vec<i64>>>>::try_from(result)?,
        &[vec![
            vec![vec![157, 158, 159, 160]],
            vec![vec![164, 165, 166, 167]],
            vec![vec![143, 144, 145, 146]]
        ]]
    );
    Ok(())
}

#[test]
fn index_3d() -> Result<()> {
    let values: Vec<i64> = (0..24).collect();
    let tensor = tch::Tensor::of_slice(&values).view((2, 3, 4));
    assert_eq!(i64::try_from(tensor.i((0, 0, 0)))?, 0);
    assert_eq!(i64::try_from(tensor.i((1, 0, 0)))?, 12);
    assert_eq!(Vec::<i64>::try_from(tensor.i((0..2, 0, 0)))?, &[0, 12]);
    Ok(())
}
