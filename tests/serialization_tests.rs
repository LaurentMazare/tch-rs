use tch::{Kind, Tensor};

struct TmpFile(std::path::PathBuf);

impl TmpFile {
    fn create(base: &str) -> TmpFile {
        let filename = std::env::temp_dir().join(format!(
            "tch-{}-{}-{:?}",
            base,
            std::process::id(),
            std::thread::current().id(),
        ));
        TmpFile(filename)
    }
}

impl std::convert::AsRef<std::path::Path> for TmpFile {
    fn as_ref(&self) -> &std::path::Path {
        self.0.as_path()
    }
}

impl Drop for TmpFile {
    fn drop(&mut self) {
        std::fs::remove_file(&self.0).unwrap()
    }
}

#[test]
fn save_and_load() {
    let tmp_file = TmpFile::create("save-and-load");
    let vec = [3.0, 1.0, 4.0, 1.0, 5.0].to_vec();
    let t1 = Tensor::of_slice(&vec);
    t1.save(&tmp_file).unwrap();
    let t2 = Tensor::load(&tmp_file).unwrap();
    assert_eq!(Vec::<f64>::from(&t2), vec)
}

#[test]
fn save_to_stream_and_load() {
    let tmp_file = TmpFile::create("write-stream");
    let vec = [3.0, 1.0, 4.0, 1.0, 5.0].to_vec();
    let t1 = Tensor::of_slice(&vec);
    t1.save_to_stream(std::fs::File::create(&tmp_file).unwrap()).unwrap();
    let t2 = Tensor::load(&tmp_file).unwrap();
    assert_eq!(Vec::<f64>::from(&t2), vec)
}

#[test]
fn save_and_load_from_stream() {
    let tmp_file = TmpFile::create("read-stream");
    let vec = [3.0, 1.0, 4.0, 1.0, 5.0].to_vec();
    let t1 = Tensor::of_slice(&vec);
    t1.save(&tmp_file).unwrap();
    let reader = std::io::BufReader::new(std::fs::File::open(&tmp_file).unwrap());
    let t2 = Tensor::load_from_stream(reader).unwrap();
    assert_eq!(Vec::<f64>::from(&t2), vec)
}

#[test]
fn save_and_load_multi() {
    let tmp_file = TmpFile::create("save-and-load-multi");
    let pi = Tensor::of_slice(&[3.0, 1.0, 4.0, 1.0, 5.0]);
    let e = Tensor::of_slice(&[2, 7, 1, 8, 2, 8, 1, 8, 2, 8, 4, 6]);
    Tensor::save_multi(&[(&"pi", &pi), (&"e", &e)], &tmp_file).unwrap();
    let named_tensors = Tensor::load_multi(&tmp_file).unwrap();
    assert_eq!(named_tensors.len(), 2);
    assert_eq!(named_tensors[0].0, "pi");
    assert_eq!(named_tensors[1].0, "e");
    assert_eq!(i64::from(&named_tensors[1].1.sum(tch::Kind::Float)), 57);
}

#[test]
fn save_to_stream_and_load_multi() {
    let tmp_file = TmpFile::create("save-to-stream-and-load-multi");
    let pi = Tensor::of_slice(&[3.0, 1.0, 4.0, 1.0, 5.0]);
    let e = Tensor::of_slice(&[2, 7, 1, 8, 2, 8, 1, 8, 2, 8, 4, 6]);
    Tensor::save_multi_to_stream(
        &[(&"pi", &pi), (&"e", &e)],
        std::fs::File::create(&tmp_file).unwrap(),
    )
    .unwrap();
    let named_tensors = Tensor::load_multi(&tmp_file).unwrap();
    assert_eq!(named_tensors.len(), 2);
    assert_eq!(named_tensors[0].0, "pi");
    assert_eq!(named_tensors[1].0, "e");
    assert_eq!(i64::from(&named_tensors[1].1.sum(tch::Kind::Float)), 57);
}

#[test]
fn save_and_load_multi_from_stream() {
    let tmp_file = TmpFile::create("save-and-load-multi-from-stream");
    let pi = Tensor::of_slice(&[3.0, 1.0, 4.0, 1.0, 5.0]);
    let e = Tensor::of_slice(&[2, 7, 1, 8, 2, 8, 1, 8, 2, 8, 4, 6]);
    Tensor::save_multi(&[(&"pi", &pi), (&"e", &e)], &tmp_file).unwrap();
    let reader = std::io::BufReader::new(std::fs::File::open(&tmp_file).unwrap());
    let named_tensors = Tensor::load_multi_from_stream(reader).unwrap();
    assert_eq!(named_tensors.len(), 2);
    assert_eq!(named_tensors[0].0, "pi");
    assert_eq!(named_tensors[1].0, "e");
    assert_eq!(i64::from(&named_tensors[1].1.sum(tch::Kind::Float)), 57);
}

#[test]
fn save_and_load_npz() {
    let tmp_file = TmpFile::create("save-and-load-npz");
    let pi = Tensor::of_slice(&[3.0, 1.0, 4.0, 1.0, 5.0]);
    let e = Tensor::of_slice(&[2, 7, 1, 8, 2, 8, 1, 8, 2, 8, 4, 6]);
    Tensor::write_npz(&[(&"pi", &pi), (&"e", &e)], &tmp_file).unwrap();
    let named_tensors = Tensor::read_npz(&tmp_file).unwrap();
    assert_eq!(named_tensors.len(), 2);
    assert_eq!(named_tensors[0].0, "pi");
    assert_eq!(named_tensors[1].0, "e");
    assert_eq!(i64::from(&named_tensors[1].1.sum(tch::Kind::Float)), 57);
}

#[test]
fn save_and_load_npz_half() {
    let tmp_file = TmpFile::create("save-and-load-npz-half");
    let pi = Tensor::of_slice(&[3.0, 1.0, 4.0, 1.0, 5.0]).to_dtype(Kind::Half, true, false);
    let e =
        Tensor::of_slice(&[2, 7, 1, 8, 2, 8, 1, 8, 2, 8, 4, 6]).to_dtype(Kind::Half, true, false);
    Tensor::write_npz(&[(&"pi", &pi), (&"e", &e)], &tmp_file).unwrap();
    let named_tensors = Tensor::read_npz(&tmp_file).unwrap();
    assert_eq!(named_tensors.len(), 2);
    assert_eq!(named_tensors[0].0, "pi");
    assert_eq!(named_tensors[1].0, "e");
    assert_eq!(i64::from(&named_tensors[1].1.sum(tch::Kind::Float)), 57);
}

#[test]
fn save_and_load_npz_byte() {
    let tmp_file = TmpFile::create("save-and-load-npz-byte");
    let pi = Tensor::of_slice(&[3.0, 1.0, 4.0, 1.0, 5.0]).to_dtype(Kind::Int8, true, false);
    let e =
        Tensor::of_slice(&[2, 7, 1, 8, 2, 8, 1, 8, 2, 8, 4, 6]).to_dtype(Kind::Int8, true, false);
    Tensor::write_npz(&[(&"pi", &pi), (&"e", &e)], &tmp_file).unwrap();
    let named_tensors = Tensor::read_npz(&tmp_file).unwrap();
    assert_eq!(named_tensors.len(), 2);
    assert_eq!(named_tensors[0].0, "pi");
    assert_eq!(named_tensors[1].0, "e");
    assert_eq!(i8::from(&named_tensors[1].1.sum(tch::Kind::Int8)), 57);
}

#[test]
fn save_and_load_npy() {
    let tmp_file = TmpFile::create("save-and-load-npy");
    let pi = Tensor::of_slice(&[3.0, 1.0, 4.0, 1.0, 5.0, 9.0]);
    pi.write_npy(&tmp_file).unwrap();
    let pi = Tensor::read_npy(&tmp_file).unwrap();
    assert_eq!(Vec::<f64>::from(&pi), [3.0, 1.0, 4.0, 1.0, 5.0, 9.0]);
    let pi = pi.reshape(&[3, 1, 2]);
    pi.write_npy(&tmp_file).unwrap();
    let pi = Tensor::read_npy(&tmp_file).unwrap();
    assert_eq!(pi.size(), [3, 1, 2]);
    assert_eq!(Vec::<f64>::from(pi.flatten(0, -1)), [3.0, 1.0, 4.0, 1.0, 5.0, 9.0]);
}
