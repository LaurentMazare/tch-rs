use tch::{IValue, Kind, Tensor};

#[test]
fn jit() {
    let x = Tensor::of_slice(&[3, 1, 4, 1, 5]).to_kind(Kind::Float);
    let y = Tensor::of_slice(&[7]).to_kind(Kind::Float);
    // The JIT module is created in create_jit_models.py
    let foo = tch::CModule::load("tests/foo.pt").unwrap();
    let result = foo.forward_ts(&[&x, &y]).unwrap();
    let expected = x * 2.0 + y + 42.0;
    assert_eq!(Vec::<f64>::from(&result), Vec::<f64>::from(&expected));
}

#[test]
fn jit1() {
    let foo = tch::CModule::load("tests/foo1.pt").unwrap();
    let result = foo
        .forward_ts(&[Tensor::from(42), Tensor::from(1337)])
        .unwrap();
    assert_eq!(i64::from(&result), 1421);
}

#[test]
fn jit2() {
    let foo = tch::CModule::load("tests/foo2.pt").unwrap();
    let result = foo
        .forward_is(&[
            IValue::Tensor(Tensor::from(42)),
            IValue::Tensor(Tensor::from(1337)),
        ])
        .unwrap();
    let expected1 = Tensor::from(1421);
    let expected2 = Tensor::from(-1295);
    assert_eq!(
        result,
        IValue::Tuple(vec![IValue::Tensor(expected1), IValue::Tensor(expected2)])
    )
}

#[test]
fn jit3() {
    let foo = tch::CModule::load("tests/foo3.pt").unwrap();
    let xs = Tensor::of_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let result = foo.forward_ts(&[xs]).unwrap();
    assert_eq!(f64::from(&result), 120.0);
}
