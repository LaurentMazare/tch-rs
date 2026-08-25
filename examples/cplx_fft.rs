#[cfg(feature = "complex")]
fn main() {
    use std::f64::consts::PI;
    use tch::*;

    let len = 32;
    let freq = 0.2;

    let n = Tensor::arange(len, (Kind::Float, Device::Cpu));
    let theta = &n * (2.0 * PI * freq);
    let real = theta.cos();
    let imag = theta.sin();
    let sinusoid = Tensor::complex(&real, &imag);
    let fft_bins = sinusoid.fft_fft(len, -1, "backward");
    let psd = fft_bins.abs().pow_tensor_scalar(1.0) / len as f64;
    let psd: Vec<f32> = psd.try_into().unwrap();
    let max_idx = psd.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).map(|(idx, _)| idx).unwrap();
    println!("Frequency: approx = {}, actual = {}", max_idx as f32 / len as f32, freq);
}