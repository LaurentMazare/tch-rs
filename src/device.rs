// This is more a device type than a device.
// If needed we should add the device index.
#[derive(Debug, Copy, Clone)]
pub enum Device {
    Cpu,
    Cuda,
}

pub enum Cpu {}
impl Cpu {
    pub fn get_num_threads() -> i64 {
        super::device_wrapper::get_num_threads()
    }

    pub fn set_num_threads(n: i64) {
        super::device_wrapper::set_num_threads(n)
    }
}

pub enum Cuda {}
impl Cuda {
    pub fn device_count() -> i64 {
        super::device_wrapper::cuda_device_count()
    }

    pub fn is_available() -> bool {
        super::device_wrapper::cuda_is_available()
    }

    pub fn cudnn_is_available() -> bool {
        super::device_wrapper::cudnn_is_available()
    }

    pub fn cudnn_set_benchmark(b: bool) {
        super::device_wrapper::set_benchmark_cudnn(b)
    }
}

impl Device {
    pub(crate) fn c_int(&self) -> libc::c_int {
        match self {
            Device::Cpu => 0,
            Device::Cuda => 1,
        }
    }

    pub(crate) fn of_c_int(v: libc::c_int) -> Self {
        match v {
            0 => Device::Cpu,
            1 => Device::Cuda,
            _ => panic!("unexpected device {}", v),
        }
    }

    pub fn cuda_if_available() -> Device {
        if Cuda::is_available() {
            Device::Cuda
        } else {
            Device::Cpu
        }
    }
}
