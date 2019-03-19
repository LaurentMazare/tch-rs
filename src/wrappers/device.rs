//! Devices on which tensor computations are run.
//!
//! This currently represents a device type rather than a device
//! which could be problematic in a multi-GPU setting.
//! If needed, a device index should be added.

/// A torch device.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Device {
    /// The main CPU device.
    Cpu,
    /// The main GPU device.
    Cuda,
}

pub enum Cpu {}
impl Cpu {
    /// Gets the current number of allowed threads for parallel computations.
    pub fn get_num_threads() -> i64 {
        let res = unsafe_torch!({ torch_sys::atc_get_num_threads() });
        i64::from(res)
    }

    /// Sets the number of allowed threads for parallel computations.
    pub fn set_num_threads(n: i64) {
        unsafe_torch!({ torch_sys::atc_set_num_threads(n as i32) })
    }
}

pub enum Cuda {}
impl Cuda {
    /// Returns the number of GPU that can be used.
    pub fn device_count() -> i64 {
        let res = unsafe_torch!({ torch_sys::atc_cuda_device_count() });
        i64::from(res)
    }

    /// Returns true if cuda support is available.
    pub fn is_available() -> bool {
        unsafe_torch!({ torch_sys::atc_cuda_is_available() }) != 0
    }

    /// Returns true if cudnn support is available.
    pub fn cudnn_is_available() -> bool {
        unsafe_torch!({ torch_sys::atc_cudnn_is_available() }) != 0
    }

    /// Sets cudnn benchmark mode.
    ///
    /// When set cudnn will try to optimize the generators durning
    /// the first network runs and then use the optimized architecture
    /// in the following runs. This can result in significant performance
    /// improvements.
    pub fn cudnn_set_benchmark(b: bool) {
        unsafe_torch!({ torch_sys::atc_set_benchmark_cudnn(if b { 1 } else { 0 }) })
    }
}

impl Device {
    pub(super) fn c_int(self) -> libc::c_int {
        match self {
            Device::Cpu => 0,
            Device::Cuda => 1,
        }
    }

    pub(super) fn of_c_int(v: libc::c_int) -> Self {
        match v {
            0 => Device::Cpu,
            1 => Device::Cuda,
            _ => panic!("unexpected device {}", v),
        }
    }

    /// Returns a GPU device if available, else default to CPU.
    pub fn cuda_if_available() -> Device {
        if Cuda::is_available() {
            Device::Cuda
        } else {
            Device::Cpu
        }
    }
}
