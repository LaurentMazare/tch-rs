use std::convert::{TryFrom, TryInto};
use tch::{Device, Tensor};

#[test]
fn assign_ops() {
    let mut t = Tensor::of_slice(&[3, 1, 4, 1, 5]);
    t += 1;
    t *= 2;
    t -= 1;
    assert_eq!(Vec::<i64>::from(&t), [7, 3, 9, 3, 11]);
}

#[test]
fn constant_ops() {
    let mut t = Tensor::of_slice(&[7i64, 3, 9, 3, 11]);
    t = -t;
    assert_eq!(Vec::<i64>::from(&t), [-7, -3, -9, -3, -11]);
    t = 1 - t;
    assert_eq!(Vec::<i64>::from(&t), [8, 4, 10, 4, 12]);
    t = 2 * t;
    assert_eq!(Vec::<i64>::from(&t), [16, 8, 20, 8, 24]);

    let mut t = Tensor::of_slice(&[0.2f64, 0.1]);
    t = 2 / t;
    assert_eq!(Vec::<f64>::from(&t), [10.0, 20.0]);
}

#[test]
fn iter() {
    let t = Tensor::of_slice(&[7i64, 3, 9, 3, 11]);
    let v = t.iter::<i64>().unwrap().collect::<Vec<_>>();
    assert_eq!(v, [7, 3, 9, 3, 11]);
    let t = Tensor::of_slice(&[3.14, 15.926, 5.3589, 79.0]);
    let v = t.iter::<f64>().unwrap().collect::<Vec<_>>();
    assert_eq!(v, [3.14, 15.926, 5.3589, 79.0]);
}

#[test]
fn array_conversion() {
    let vec: Vec<_> = (0..6).map(|x| (x * x) as f64).collect();
    let t = Tensor::of_slice(&vec);
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
    let t = Tensor::of_slice(&[3.0, 1.0, 4.0, 1.0, 5.0]);
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
    let t = Tensor::of_slice(&[13.0, 37.0]);
    let t = Tensor::cat(&[&t, &t, &t], 0);
    assert_eq!(t.size(), [6]);
    assert_eq!(Vec::<f64>::from(&t), [13.0, 37.0, 13.0, 37.0, 13.0, 37.0]);

    let t = Tensor::of_slice(&[13.0, 37.0]);
    let t = Tensor::stack(&[&t, &t, &t], 0);
    assert_eq!(t.size(), [3, 2]);
    assert_eq!(Vec::<f64>::from(&t), [13.0, 37.0, 13.0, 37.0, 13.0, 37.0]);

    let t = Tensor::of_slice(&[13.0, 37.0]);
    let t = Tensor::stack(&[&t, &t, &t], 1);
    assert_eq!(t.size(), [2, 3]);
    assert_eq!(Vec::<f64>::from(&t), [13.0, 13.0, 13.0, 37.0, 37.0, 37.0]);
}

#[test]
fn save_and_load() {
    let filename = std::env::temp_dir().join(format!("tch-{}", std::process::id()));
    let vec = [3.0, 1.0, 4.0, 1.0, 5.0].to_vec();
    let t1 = Tensor::of_slice(&vec);
    t1.save(&filename).unwrap();
    let t2 = Tensor::load(&filename).unwrap();
    assert_eq!(Vec::<f64>::from(&t2), vec)
}

#[test]
fn save_and_load_multi() {
    let filename = std::env::temp_dir().join(format!("tch2-{}", std::process::id()));
    let pi = Tensor::of_slice(&[3.0, 1.0, 4.0, 1.0, 5.0]);
    let e = Tensor::of_slice(&[2, 7, 1, 8, 2, 8, 1, 8, 2, 8, 4, 6]);
    Tensor::save_multi(&[(&"pi", &pi), (&"e", &e)], &filename).unwrap();
    let named_tensors = Tensor::load_multi(&filename).unwrap();
    assert_eq!(named_tensors.len(), 2);
    assert_eq!(named_tensors[0].0, "pi");
    assert_eq!(named_tensors[1].0, "e");
    assert_eq!(i64::from(&named_tensors[1].1.sum()), 57);
}

#[test]
fn onehot() {
    let xs = Tensor::of_slice(&[0, 1, 2, 3]);
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
    let xs = Tensor::of_slice(&[0, 1, 2, 3]);
    let ys = Tensor::of_slice(&[0, 1, 2, 3, 4]);
    assert!(xs.f_eq1(&ys).is_err())
}

#[test]
fn chunk() {
    let xs = Tensor::of_slice(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    let tensors = xs.chunk(3, 0);
    assert_eq!(tensors.len(), 3);
    assert_eq!(Vec::<i64>::from(&tensors[0]), vec![0, 1, 2, 3]);
    assert_eq!(Vec::<i64>::from(&tensors[1]), vec![4, 5, 6, 7]);
    assert_eq!(Vec::<i64>::from(&tensors[2]), vec![8, 9]);
}

#[test]
fn broadcast() {
    let xs = Tensor::of_slice(&[4, 5, 3]);
    let ys = Tensor::from(42);
    let tensors = Tensor::broadcast_tensors(&[xs, ys]);
    assert_eq!(tensors.len(), 2);
    assert_eq!(Vec::<i64>::from(&tensors[0]), vec![4, 5, 3]);
    assert_eq!(Vec::<i64>::from(&tensors[1]), vec![42, 42, 42]);
}

#[test]
fn eq() {
    let t = Tensor::of_slice(&[3, 1, 4, 1, 5]);
    let u = &t + 1 - 1;
    assert_eq!(t, u);
    assert!(t == u);
    assert!(t != u - 1);

    let t = Tensor::of_slice(&[3.14]);
    let u = Tensor::from(3.14);
    // The tensor shape is important for equality.
    assert!(t != u);
    assert!(t.size() != u.size());

    let u = u.reshape(&[1]);
    assert_eq!(t, u);
    assert!(t == u);
    assert!(t != u - 1)
}

#[test]
fn values_at_index() {
    let t = Tensor::from(42);
    assert_eq!(t.int64_value(&[]), 42);
    assert_eq!(t.double_value(&[]), 42.0);
    assert!(t.f_int64_value(&[0]).is_err());
    assert!(t.f_double_value(&[0]).is_err());
}

#[test]
fn into_ndarray_f64() {
    let tensor = Tensor::of_slice(&[1., 2., 3., 4.]).reshape(&[2, 2]);
    let nd: ndarray::ArrayD<f64> = (&tensor).try_into().unwrap();
    assert_eq!(Vec::<f64>::from(tensor).as_slice(), nd.as_slice().unwrap());
}

#[test]
fn into_ndarray_i64() {
    let tensor = Tensor::of_slice(&[1, 2, 3, 4]).reshape(&[2, 2]);
    let nd: ndarray::ArrayD<i64> = (&tensor).try_into().unwrap();
    assert_eq!(Vec::<i64>::from(tensor).as_slice(), nd.as_slice().unwrap());
}

#[test]
fn from_ndarray_f64() {
    let nd = ndarray::arr2(&[[1f64, 2.], [3., 4.]]);
    let tensor = Tensor::try_from(nd.clone()).unwrap();
    assert_eq!(Vec::<f64>::from(tensor).as_slice(), nd.as_slice().unwrap());
}

#[test]
fn from_ndarray_i64() {
    let nd = ndarray::arr2(&[[1i64, 2], [3, 4]]);
    let tensor = Tensor::try_from(nd.clone()).unwrap();
    assert_eq!(Vec::<i64>::from(tensor).as_slice(), nd.as_slice().unwrap());
}

#[test]
fn test_device() {
    let x = Tensor::from(1);
    assert_eq!(x.device(), Device::Cpu);
    let x = Tensor::from(1).to_device(Device::Cpu);
    assert_eq!(x.device(), Device::Cpu);
    if tch::Cuda::device_count() > 0 {
        let x = Tensor::from(1).to_device(Device::Cuda(0));
        assert_eq!(x.device(), Device::Cuda(0));
        let x = Tensor::from(1)
            .to_device(Device::Cuda(0))
            .to_device(Device::Cpu);
        assert_eq!(x.device(), Device::Cpu);
    }
}
