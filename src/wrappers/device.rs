//! Devices on which tensor computations are run.
use crate::TchError;
use torch_sys::{C_cuda_stream, C_cuda_stream_guard};

/// A torch device.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Device {
    /// The main CPU device.
    Cpu,
    /// The main GPU device.
    Cuda(usize),
}

/// A CUDA stream object.
pub struct CudaStream {
    pub(super) c_cuda_stream: *mut C_cuda_stream,
}

impl CudaStream {
    /// Get a Cuda stream object from the pool for a given device.
    pub fn from_pool(high_priority: bool, device: Option<usize>) -> Result<Self, TchError> {
        let high_priority = if high_priority { 1 } else { 0 };
        let device = match device {
            Some(d) => d as i32,
            None => -1,
        };
        let c_cuda_stream =
            unsafe_torch_err!(torch_sys::get_stream_from_pool(high_priority, device));

        if c_cuda_stream.is_null() {
            return Err(TchError::Torch("not available".to_string()));
        }

        Ok(Self { c_cuda_stream })
    }
}

impl Drop for CudaStream {
    fn drop(&mut self) {
        unsafe_torch!(torch_sys::delete_stream(self.c_cuda_stream))
    }
}

/// A CUDA stream guard object.
pub struct CudaStreamGuard {
    pub(super) c_cuda_stream_guard: *mut C_cuda_stream_guard,
}

impl CudaStreamGuard {
    /// Activate a Cuda stream guard for the provided stream, this guard is
    /// active until the end of the scope or until another guard is initailized.
    pub fn new(stream: CudaStream) -> Result<Self, TchError> {
        let c_cuda_stream_guard =
            unsafe_torch_err!(torch_sys::get_stream_guard(stream.c_cuda_stream));

        if c_cuda_stream_guard.is_null() {
            return Err(TchError::Torch("not available".to_string()));
        }

        Ok(Self {
            c_cuda_stream_guard,
        })
    }
}

impl Drop for CudaStreamGuard {
    fn drop(&mut self) {
        unsafe_torch!(torch_sys::delete_stream_guard(self.c_cuda_stream_guard))
    }
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
