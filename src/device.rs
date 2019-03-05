/// Devices on which tensor computations are run.
///
/// This currently represents a device type rather than a device
/// which could be problematic in a multi-GPU setting.
/// If needed, a device index should be added.

/// A torch device.
#[derive(Debug, Copy, Clone)]
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
        super::device_wrapper::get_num_threads()
    }

    /// Sets the number of allowed threads for parallel computations.
    pub fn set_num_threads(n: i64) {
        super::device_wrapper::set_num_threads(n)
    }
}

pub enum Cuda {}
impl Cuda {
    /// Returns the number of GPU that can be used.
    pub fn device_count() -> i64 {
        super::device_wrapper::cuda_device_count()
    }

    /// Returns true if cuda support is available.
    pub fn is_available() -> bool {
        super::device_wrapper::cuda_is_available()
    }

    /// Returns true if cudnn support is available.
    pub fn cudnn_is_available() -> bool {
        super::device_wrapper::cudnn_is_available()
    }

    /// Sets cudnn benchmark mode.
    ///
    /// When set cudnn will try to optimize the generators durning
    /// the first network runs and then use the optimized architecture
    /// in the following runs. This can result in significant performance
    /// improvements.
    pub fn cudnn_set_benchmark(b: bool) {
        super::device_wrapper::set_benchmark_cudnn(b)
    }
}

impl Device {
    pub(crate) fn c_int(self) -> libc::c_int {
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

    /// Returns a GPU device if available, else default to CPU.
    pub fn cuda_if_available() -> Device {
        if Cuda::is_available() {
            Device::Cuda
        } else {
            Device::Cpu
        }
    }
}
