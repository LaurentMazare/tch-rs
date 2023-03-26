use tch::{kind, Tensor};

fn grad_example() {
    let mut x = Tensor::from(2.0).set_requires_grad(true);
    let y = &x * &x + &x + 36;
    println!("{}", y.double_value(&[]));
    x.zero_grad();
    y.backward();
    let mut dy_over_dx = x.grad();
    dy_over_dx.print();
    let t2 = dy_over_dx.clamp_(1.23, 1.24);
    dy_over_dx.print();
    t2.print();
    println!("Grad {}", dy_over_dx.double_value(&[]));
}

fn main() {
    //println!("Cuda available: {}", tch::Cuda::is_available());
    //println!("Cudnn available: {}", tch::Cuda::cudnn_is_available());
    //TODO: Fix this for ios and non-ios.
    let device = tch::Device::Mps;
    let t = Tensor::of_slice(&[3, 1, 4, 1, 5]).to(device);
    t.print();
    let t = Tensor::randn(&[5, 4], kind::FLOAT_CPU);
    t.print();
    (&t + 1.5).print();
    (&t + 2.5).print();
    let mut t = Tensor::of_slice(&[1.1f32, 2.1, 3.1]);
    t += 42;
    t.print();
    println!("{:?} {}", t.size(), t.double_value(&[1]));
    grad_example();
    println!("has_mps: {}", tch::utils::has_mps());
    println!("version_cudnn: {}", tch::utils::version_cudnn());
    println!("version_cudart: {}", tch::utils::version_cudart());
}
