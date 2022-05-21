use std::convert::TryFrom;
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
fn jit_data() {
    let x = Tensor::of_slice(&[3, 1, 4, 1, 5]).to_kind(Kind::Float);
    let y = Tensor::of_slice(&[7]).to_kind(Kind::Float);
    let mut file = std::fs::File::open("tests/foo.pt").unwrap();
    let foo = tch::CModule::load_data(&mut file).unwrap();
    let result = foo.forward_ts(&[&x, &y]).unwrap();
    let expected = x * 2.0 + y + 42.0;
    assert_eq!(Vec::<f64>::from(&result), Vec::<f64>::from(&expected));
}

#[test]
fn jit1() {
    let foo = tch::CModule::load("tests/foo1.pt").unwrap();
    let result = foo.forward_ts(&[Tensor::from(42), Tensor::from(1337)]).unwrap();
    assert_eq!(i64::from(&result), 1421);
    let result = foo.method_ts("forward", &[Tensor::from(42), Tensor::from(1337)]).unwrap();
    assert_eq!(i64::from(&result), 1421);
}

#[test]
fn jit2() {
    let foo = tch::CModule::load("tests/foo2.pt").unwrap();
    let result = foo
        .forward_is(&[IValue::from(Tensor::from(42)), IValue::from(Tensor::from(1337))])
        .unwrap();
    let expected1 = Tensor::from(1421);
    let expected2 = Tensor::from(-1295);
    assert_eq!(result, IValue::from((expected1, expected2)));
    // Destructure the tuple, using an option.
    let (v1, v2) = <(Tensor, Option<Tensor>)>::try_from(result).unwrap();
    assert_eq!(i64::from(v1), 1421);
    assert_eq!(i64::from(v2.unwrap()), -1295);
    let result = foo
        .method_is("forward", &[IValue::from(Tensor::from(42)), IValue::from(Tensor::from(1337))])
        .unwrap();
    let expected1 = Tensor::from(1421);
    let expected2 = Tensor::from(-1295);
    assert_eq!(result, IValue::from((expected1, expected2)));
    let (v1, v2) = <(Tensor, Tensor)>::try_from(result).unwrap();
    assert_eq!(i64::from(v1), 1421);
    assert_eq!(i64::from(v2), -1295);
}

#[test]
fn jit3() {
    let foo = tch::CModule::load("tests/foo3.pt").unwrap();
    let xs = Tensor::of_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let result = foo.forward_ts(&[xs]).unwrap();
    assert_eq!(f64::from(&result), 120.0);
}

#[test]
fn jit4() {
    let foo = tch::CModule::load("tests/foo4.pt").unwrap();
    let result = foo.forward_is(&[IValue::from((2.0, 3.0, 4))]).unwrap();
    assert_eq!(result, 14.0.into());
    let v = f64::try_from(result).unwrap();
    assert_eq!(v, 14.0);
    let named_parameters = foo.named_parameters().unwrap();
    assert_eq!(named_parameters, vec![]);
}

#[test]
fn profiling_mode() {
    assert_eq!(tch::jit::get_profiling_mode(), true);
    tch::jit::set_profiling_mode(false);
    assert_eq!(tch::jit::get_profiling_mode(), false);
    tch::jit::set_profiling_mode(true);
    assert_eq!(tch::jit::get_profiling_mode(), true);
}

#[test]
fn jit5() {
    let foo = tch::CModule::load("tests/foo5.pt").unwrap();
    let result = foo
        .forward_is(&[IValue::StringList(vec![
            "foo".to_string(),
            "bar".to_string(),
            "foobar".to_string(),
        ])])
        .unwrap();
    assert_eq!(
        result,
        IValue::from(vec![IValue::from("fo"), IValue::from("ba"), IValue::from("fooba")])
    );
    // Destructuring of ivalue.
    let (v1, v2, v3) = <(String, String, String)>::try_from(result).unwrap();
    assert_eq!(v1, "fo");
    assert_eq!(v2, "ba");
    assert_eq!(v3, "fooba");
}

#[test]
fn jit6() {
    let foo = tch::CModule::load("tests/foo6.pt").unwrap();
    let xs = Tensor::of_slice(&[3.0, 4.0, 5.0]);
    let result = foo.forward_is(&[IValue::Tensor(xs)]).unwrap();

    if let IValue::Object(obj) = result {
        let result = obj.method_is::<IValue>("y", &[]).unwrap();
        assert_eq!(result, IValue::Tensor(Tensor::of_slice(&[6.0, 8.0, 10.0])));
    } else {
        panic!("expected output to be an object");
    }
}

#[test]
fn create_traced() {
    let mut closure = |inputs: &[Tensor]| {
        let v1 = inputs[0].shallow_clone();
        let v2 = inputs[1].shallow_clone();
        vec![v1 + v2]
    };
    let modl = tch::CModule::create_by_tracing(
        "MyModule",
        "MyFn",
        &[Tensor::from(0.0), Tensor::from(1.0)],
        &mut closure,
    )
    .unwrap();
    let filename = std::env::temp_dir().join(format!("tch-modl-{}", std::process::id()));
    modl.save(&filename).unwrap();
    let modl = tch::CModule::load(&filename).unwrap();
    let xs = Tensor::of_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let ys = Tensor::of_slice(&[41.0, 1335.0, 0.1415, 4.0, 5.0]);
    let result = modl.method_ts("MyFn", &[xs, ys]).unwrap();
    assert_eq!(Vec::<f64>::from(&result), [42.0, 1337.0, 3.1415, 8.0, 10.0])
}

// https://github.com/LaurentMazare/tch-rs/issues/475
#[test]
fn jit_double_free() {
    let foo = tch::CModule::load("tests/foo7.pt").unwrap();
    let input = foo.method_is(
        "make_input_object",
        &[
            &Tensor::of_slice(&[1_f32, 2_f32, 3_f32]).into(),
            &Tensor::of_slice(&[4_f32, 5_f32, 6_f32]).into(),
        ],
    );
    let result = foo.method_is("add_them", &[&input.unwrap()]);
    let result = match result.unwrap() {
        IValue::Tensor(tensor) => tensor,
        result => panic!("expected a tensor got {:?}", result),
    };
    assert_eq!(Vec::<f64>::from(&result), [5.0, 7.0, 9.0])
}
