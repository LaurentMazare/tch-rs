use tch::{Kind, Tensor};

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
    assert_eq!(i64::from(&named_tensors[1].1.sum(tch::Kind::Float)), 57);
}

#[test]
fn save_and_load_npz() {
    let filename = std::env::temp_dir().join(format!("tch3-{}.npz", std::process::id()));
    let pi = Tensor::of_slice(&[3.0, 1.0, 4.0, 1.0, 5.0]);
    let e = Tensor::of_slice(&[2, 7, 1, 8, 2, 8, 1, 8, 2, 8, 4, 6]);
    Tensor::write_npz(&[(&"pi", &pi), (&"e", &e)], &filename).unwrap();
    let named_tensors = Tensor::read_npz(&filename).unwrap();
    assert_eq!(named_tensors.len(), 2);
    assert_eq!(named_tensors[0].0, "pi");
    assert_eq!(named_tensors[1].0, "e");
    assert_eq!(i64::from(&named_tensors[1].1.sum(tch::Kind::Float)), 57);
}

#[test]
fn save_and_load_npz_half() {
    let filename = std::env::temp_dir().join(format!("tch4-{}.npz", std::process::id()));
    let pi = Tensor::of_slice(&[3.0, 1.0, 4.0, 1.0, 5.0]).to2(Kind::Half, true, false);
    let e = Tensor::of_slice(&[2, 7, 1, 8, 2, 8, 1, 8, 2, 8, 4, 6]).to2(Kind::Half, true, false);
    Tensor::write_npz(&[(&"pi", &pi), (&"e", &e)], &filename).unwrap();
    let named_tensors = Tensor::read_npz(&filename).unwrap();
    assert_eq!(named_tensors.len(), 2);
    assert_eq!(named_tensors[0].0, "pi");
    assert_eq!(named_tensors[1].0, "e");
    assert_eq!(i64::from(&named_tensors[1].1.sum(tch::Kind::Float)), 57);
}
