use anyhow::Result;
use approx::assert_abs_diff_eq;
use std::convert::TryFrom;
use tch::{IValue, Kind, Tensor};

#[test]
fn jit() -> Result<()> {
    let x = Tensor::of_slice(&[3, 1, 4, 1, 5]).to_kind(Kind::Float);
    let y = Tensor::of_slice(&[7]).to_kind(Kind::Float);
    // The JIT module is created in create_jit_models.py
    let foo = tch::CModule::load("tests/foo.pt").unwrap();
    let result = foo.forward_ts(&[&x, &y]).unwrap();
    let expected = x * 2.0 + y + 42.0;
    assert_eq!(
        Vec::<f32>::try_from(&result)?,
        Vec::<f32>::try_from(&expected)?
    );
    Ok(())
}

#[test]
fn jit_data() -> Result<()> {
    let x = Tensor::of_slice(&[3, 1, 4, 1, 5]).to_kind(Kind::Float);
    let y = Tensor::of_slice(&[7]).to_kind(Kind::Float);
    let mut file = std::fs::File::open("tests/foo.pt").unwrap();
    let foo = tch::CModule::load_data(&mut file).unwrap();
    let result = foo.forward_ts(&[&x, &y]).unwrap();
    let expected = x * 2.0 + y + 42.0;
    assert_eq!(
        Vec::<f32>::try_from(&result)?,
        Vec::<f32>::try_from(&expected)?
    );
    Ok(())
}

#[test]
fn jit1() -> Result<()> {
    let foo = tch::CModule::load("tests/foo1.pt").unwrap();
    let result = foo
        .forward_ts(&[Tensor::from(42), Tensor::from(1337)])
        .unwrap();
    assert_abs_diff_eq!(f32::try_from(&result)?, 1421.0);
    Ok(())
}

#[test]
fn jit2() -> Result<()> {
    let foo = tch::CModule::load("tests/foo2.pt").unwrap();
    let result = foo.forward_is(&[
        IValue::from(Tensor::from(42)),
        IValue::from(Tensor::from(1337)),
    ])?;
    let expected1 = Tensor::from(1421);
    let expected2 = Tensor::from(-1295);
    assert_eq!(result, IValue::from((expected1, expected2)));
    Ok(())
}

#[test]
fn jit3() -> Result<()> {
    let foo = tch::CModule::load("tests/foo3.pt").unwrap();
    let xs = Tensor::of_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let result = foo.forward_ts(&[xs]).unwrap();
    assert_eq!(f64::try_from(&result)?, 120.0);
    Ok(())
}

#[test]
fn jit4() {
    let foo = tch::CModule::load("tests/foo4.pt").unwrap();
    let result = foo.forward_is(&[IValue::from((2.0, 3.0, 4))]).unwrap();
    assert_eq!(result, 14.0.into());
}

/*
#[test]
fn jit5() {
    let foo = tch::CModule::load("tests/foo5.pt").unwrap();
    let result = foo
        .forward_is(&[IValue::GenericList(vec![
            IValue::from("foo"),
            IValue::from("bar"),
            IValue::from("foobar"),
        ])])
        .unwrap();
    assert_eq!(
        result,
        IValue::from(vec![
            IValue::from("fo"),
            IValue::from("ba"),
            IValue::from("fooba")
        ])
    );
}
*/
