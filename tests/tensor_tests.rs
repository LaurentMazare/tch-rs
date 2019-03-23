use tch::Tensor;

#[test]
fn assign_ops() {
    let mut t = Tensor::int_vec(&[3, 1, 4, 1, 5]);
    t += 1;
    t *= 2;
    t -= 1;
    assert_eq!(Vec::<i64>::from(&t), [7, 3, 9, 3, 11]);
    t = -t;
    assert_eq!(Vec::<i64>::from(&t), [-7, -3, -9, -3, -11]);
}

#[test]
fn array_conversion() {
    let vec: Vec<_> = (0..6).map(|x| (x * x) as f64).collect();
    let t = Tensor::float_vec(&vec);
    assert_eq!(Vec::<f64>::from(&t), [0.0, 1.0, 4.0, 9.0, 16.0, 25.0]);
    let t = t.view(&[3, 2]);
    assert_eq!(
        Vec::<Vec<f64>>::from(&t),
        [[0.0, 1.0], [4.0, 9.0], [16.0, 25.0]]
    );
    let t = t.view(&[2, 3]);
    assert_eq!(
        Vec::<Vec<f64>>::from(&t),
        [[0.0, 1.0, 4.0], [9.0, 16.0, 25.0]]
    )
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
fn grad_grad() {
    // Compute a second order derivative using run_backward.
    let mut x = Tensor::from(42.0).set_requires_grad(true);
    let y = &x * &x * &x + &x + &x * &x;
    x.zero_grad();
    let dy_over_dx = Tensor::run_backward(&[y], &[&x], true, true);
    assert_eq!(dy_over_dx.len(), 1);
    let dy_over_dx = &dy_over_dx[0];
    dy_over_dx.backward();
    let dy_over_dx2 = x.grad();
    assert_eq!(f64::from(&dy_over_dx2), 254.0);
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
fn onehot() {
    let xs = Tensor::int_vec(&[0, 1, 2, 3]);
    let onehot = xs.onehot(4);
    assert_eq!(
        Vec::<f64>::from(&onehot),
        vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    );
    assert_eq!(onehot.size(), vec![4, 4])
}

#[test]
fn fallible() {
    // Try to compare two tensors with incompatible dimensions and check that this returns an
    // error.
    let xs = Tensor::int_vec(&[0, 1, 2, 3]);
    let ys = Tensor::int_vec(&[0, 1, 2, 3, 4]);
    assert!(xs.f_eq1(&ys).is_err())
}
