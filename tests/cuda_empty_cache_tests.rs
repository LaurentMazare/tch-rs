use tch::{Device, Kind, Tensor};
use torch_sys::c10_cuda::empty_cuda_cache;

#[test]
fn cuda_empty_cache() {
    println!("Cuda empty cache test started...");
    println!("Create tensor...");
    let tensor = Tensor::randn([1024, 1024, 1024], (Kind::Float, Device::Cpu));
    println!("Push tensor to cuda");
    let tensor_cuda = tensor.to_kind(Kind::Half).to_device(Device::Cuda(0));
    println!("Empty cuda cache");
    drop(tensor_cuda);
    let _ = empty_cuda_cache();
}
