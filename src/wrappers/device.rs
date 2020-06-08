//! Devices on which tensor computations are run.

/// A torch device.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Device {
    /// The main CPU device.
    Cpu,
    /// The main GPU device.
    Cuda(usize),
}

/// Cuda related helper functions.
pub enum Cuda {}
impl Cuda {
    /// Returns the number of GPU that can be used.
    pub fn device_count() -> i64 {
        let res = unsafe_torch!(torch_sys::atc_cuda_device_count());
        i64::from(res)
    }

    /// Returns true if cuda support is available.
    pub fn is_available() -> bool {
        unsafe_torch!(torch_sys::atc_cuda_is_available()) != 0
    }

    /// Returns true if cudnn support is available.
    pub fn cudnn_is_available() -> bool {
        unsafe_torch!(torch_sys::atc_cudnn_is_available()) != 0
    }

    /// Sets cudnn benchmark mode.
    ///
    /// When set cudnn will try to optimize the generators durning
    /// the first network runs and then use the optimized architecture
    /// in the following runs. This can result in significant performance
    /// improvements.
    pub fn cudnn_set_benchmark(b: bool) {
        unsafe_torch!(torch_sys::atc_set_benchmark_cudnn(if b { 1 } else { 0 }))
    }
}

impl Device {
    pub(super) fn c_int(self) -> libc::c_int {
        match self {
            Device::Cpu => -1,
            Device::Cuda(device_index) => device_index as libc::c_int,
        }
    }

    pub(super) fn of_c_int(v: libc::c_int) -> Self {
        match v {
            -1 => Device::Cpu,
            index if index >= 0 => Device::Cuda(index as usize),
            _ => panic!("unexpected device {}", v),
        }
    }

    /// Returns a GPU device if available, else default to CPU.
    pub fn cuda_if_available() -> Device {
        if Cuda::is_available() {
            Device::Cuda(0)
        } else {
            Device::Cpu
        }
    }

    pub fn is_cuda(self) -> bool {
        match self {
            Device::Cuda(_) => true,
            Device::Cpu => false,
        }
    }
}
