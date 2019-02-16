extern crate torchr;

fn main() {
    let t = torchr::Tensor::int_vec(&[3, 1, 4, 1, 5]);
    t.print();
    let t = torchr::Tensor::randn(&[5, 4]);
    t.print();
    (t + torchr::Scalar::from(1.5)).print();
    let t = torchr::Tensor::randn(&[5, 4, 3000, 3000, 3000, 3000, 3000]);
    t.print();
}
