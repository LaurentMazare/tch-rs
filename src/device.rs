//! Devices on which tensor computations are run.
//!
//! This currently represents a device type rather than a device
//! which could be problematic in a multi-GPU setting.
//! If needed, a device index should be added.

/// A torch device.
#[repr(u32)]
#[derive(Debug, Copy, Clone)]
pub enum Device {
    /// The main CPU device.
    Cpu,
    /// The main GPU device.
    Cuda,
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

    /// Returns a GPU device if available, else default to CPU.
    pub fn cuda_if_available() -> Self {
        if Cuda.cuda_is_available() {
            Device::Cuda
        } else {
            Device::Cpu
        }
    }
}

pub trait TorchDevice: Sized {
    fn get_num_threads(&self) -> Option<i64> {
        None
    }

    fn set_num_threads(&self, _num_threads: i64) {
        ()
    }

    fn device_count(&self) -> i64 {
        0
    }

    fn cuda_is_available(&self) -> bool {
        false
    }

    fn cudnn_is_available(&self) -> bool {
        false
    }

    fn set_benchmark_cudnn(&self, _b: bool) {
        ()
    }
}

#[derive(Debug, Copy, Clone)]
pub struct Cpu;

impl TorchDevice for Cpu {
    /// Gets the current number of allowed threads for parallel computations.
    fn get_num_threads(&self) -> Option<i64> {
        let res = unsafe_torch!({ torch_sys::atc_get_num_threads() });
        Some(res.into())
    }

    /// Sets the number of allowed threads for parallel computations.
    fn set_num_threads(&self, num_threads: i64) {
        unsafe_torch!({ torch_sys::atc_set_num_threads(num_threads as i32) });
    }
}

#[derive(Debug, Copy, Clone)]
pub struct Cuda;

impl TorchDevice for Cuda {
    /// Returns the number of GPU that can be used.
    fn device_count(&self) -> i64 {
        let res = unsafe_torch!({ torch_sys::atc_cuda_device_count() });
        res.into()
    }

    /// Returns true if cuda support is available.
    fn cuda_is_available(&self) -> bool {
        let ret = unsafe_torch!({ torch_sys::atc_cuda_is_available() });
        ret != 0
    }

    /// Returns true if cudnn support is available.
    fn cudnn_is_available(&self) -> bool {
        let ret = unsafe_torch!({ torch_sys::atc_cudnn_is_available() });
        ret != 0
    }

    /// Sets cudnn benchmark mode.
    ///
    /// When set cudnn will try to optimize the generators durning
    /// the first network runs and then use the optimized architecture
    /// in the following runs. This can result in significant performance
    /// improvements.
    fn set_benchmark_cudnn(&self, b: bool) {
        unsafe_torch!({ torch_sys::atc_set_benchmark_cudnn(b as i32) });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_attributes() {
        let cpu = Cpu;
        assert_eq!(cpu.get_num_threads(), Some(-1));
        cpu.set_num_threads(1);
        assert_eq!(cpu.get_num_threads(), Some(1));
        assert!(!cpu.cuda_is_available());
        assert!(!cpu.cudnn_is_available());
    }
}
