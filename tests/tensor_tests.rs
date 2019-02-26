use tch::nn::VarStore;
use tch::Device;
use tch::Tensor;

#[test]
fn assign_ops() {
    let mut t = Tensor::int_vec(&[3, 1, 4, 1, 5]);
    t += 1;
    t *= 2;
    t -= 1;
    assert_eq!(Vec::<i64>::from(&t), [7, 3, 9, 3, 11]);
}

#[test]
fn binary_ops() {
    let t = Tensor::float_vec(&[3.0, 1.0, 4.0, 1.0, 5.0]);
    let t = (&t * &t) + &t - 1.5;
    assert_eq!(Vec::<f64>::from(&t), [10.5, 0.5, 18.5, 0.5, 28.5]);
}

#[test]
fn grad() {
    let mut x = Tensor::from(2.0).set_requires_grad(true);
    let y = &x * &x + &x + 36;
    x.zero_grad();
    y.backward();
    let dy_over_dx = x.grad();
    assert_eq!(Vec::<f64>::from(&dy_over_dx), [5.0]);
}

#[test]
fn cat_and_stack() {
    let t = Tensor::float_vec(&[13.0, 37.0]);
    let t = Tensor::cat(&[&t, &t, &t], 0);
    assert_eq!(t.size(), [6]);
    assert_eq!(Vec::<f64>::from(&t), [13.0, 37.0, 13.0, 37.0, 13.0, 37.0]);

    let t = Tensor::float_vec(&[13.0, 37.0]);
    let t = Tensor::stack(&[&t, &t, &t], 0);
    assert_eq!(t.size(), [3, 2]);
    assert_eq!(Vec::<f64>::from(&t), [13.0, 37.0, 13.0, 37.0, 13.0, 37.0]);

    let t = Tensor::float_vec(&[13.0, 37.0]);
    let t = Tensor::stack(&[&t, &t, &t], 1);
    assert_eq!(t.size(), [2, 3]);
    assert_eq!(Vec::<f64>::from(&t), [13.0, 13.0, 13.0, 37.0, 37.0, 37.0]);
}

#[test]
fn save_and_load() {
    let filename = std::env::temp_dir().join(format!("tch-{}", std::process::id()));
    let vec = [3.0, 1.0, 4.0, 1.0, 5.0].to_vec();
    let t1 = Tensor::float_vec(&vec);
    t1.save(&filename).unwrap();
    let t2 = Tensor::load(&filename).unwrap();
    assert_eq!(Vec::<f64>::from(&t2), vec)
}

#[test]
fn save_and_load_multi() {
    let filename = std::env::temp_dir().join(format!("tch2-{}", std::process::id()));
    let pi = Tensor::float_vec(&[3.0, 1.0, 4.0, 1.0, 5.0]);
    let e = Tensor::int_vec(&[2, 7, 1, 8, 2, 8, 1, 8, 2, 8, 4, 6]);
    Tensor::save_multi(&[(&"pi", &pi), (&"e", &e)], &filename).unwrap();
    let named_tensors = Tensor::load_multi(&filename).unwrap();
    assert_eq!(named_tensors.len(), 2);
    assert_eq!(named_tensors[0].0, "pi");
    assert_eq!(named_tensors[1].0, "e");
    assert_eq!(i64::from(&named_tensors[1].1.sum()), 57);
}

#[test]
fn save_and_load_var_store() {
    let filename = std::env::temp_dir().join(format!("tch-vs-{}", std::process::id()));
    let add = |vs: &mut tch::nn::VarStore| {
        let u = vs.root().zeros("t1", &[4]);
        let v = vs.root().sub("a").sub("b").ones("t2", &[3]);
        (u, v)
    };
    let mut vs1 = VarStore::new(Device::Cpu);
    let mut vs2 = VarStore::new(Device::Cpu);
    let (mut u1, mut v1) = add(&mut vs1);
    let (u2, v2) = add(&mut vs2);
    tch::no_grad(|| {
        u1 += 42.0;
        v1 *= 2.0;
    });
    assert_eq!(f64::from(&u1.mean()), 42.0);
    assert_eq!(f64::from(&v1.mean()), 2.0);
    assert_eq!(f64::from(&u2.mean()), 0.0);
    assert_eq!(f64::from(&v2.mean()), 1.0);
    vs1.save(&filename).unwrap();
    vs2.load(&filename).unwrap();
    assert_eq!(f64::from(&u1.mean()), 42.0);
    assert_eq!(f64::from(&u2.mean()), 42.0);
    assert_eq!(f64::from(&v2.mean()), 2.0);
}
