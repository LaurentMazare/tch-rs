use tch::{Kind, Tensor};

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
