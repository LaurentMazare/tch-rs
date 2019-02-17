extern crate torchr;

fn main() {
    let t = torchr::Tensor::int_vec(&[3, 1, 4, 1, 5]);
    t.print();
    let t = torchr::Tensor::randn(&[5, 4], torchr::Kind::Float);
    t.print();
    (t + 1.5).print();
    let mut t = torchr::Tensor::float_vec(&[1.1, 2.1, 3.1]);
    t += 42;
    t.print();
    let t = torchr::Tensor::randn(&[5, 4, 3000, 3000, 3000, 3000, 3000], torchr::Kind::Float);
    t.print();
}
