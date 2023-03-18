//! CUDA Stream API.

use crate::TchError;
use libc::c_int;

pub struct CudaStream {
    c_ptr: *mut torch_sys::cuda::CCudaStream,
}

impl CudaStream {
    pub fn get_stream_from_pool(high_priority: bool, device: usize) -> Result<Self, TchError> {
        let c_ptr = unsafe_torch_err!(torch_sys::cuda::atcs_get_stream_from_pool(
            high_priority as c_int,
            device as c_int
        ));
        if c_ptr.is_null() {
            return Err(TchError::Torch(
                "CUDAStream::getStreamFromPool() returned null".to_string(),
            ));
        }
        Ok(Self { c_ptr })
    }

    pub fn get_default_stream(device: usize) -> Result<Self, TchError> {
        let c_ptr = unsafe_torch_err!(torch_sys::cuda::atcs_get_default_stream(device as c_int));
        if c_ptr.is_null() {
            return Err(TchError::Torch(
                "CUDAStream::getDefaultStream() returned null".to_string(),
            ));
        }
        Ok(Self { c_ptr })
    }

    pub fn get_current_stream(device: usize) -> Result<Self, TchError> {
        let c_ptr = unsafe_torch_err!(torch_sys::cuda::atcs_get_current_stream(device as c_int));
        if c_ptr.is_null() {
            return Err(TchError::Torch(
                "CUDAStream::getStreamFromPool() returned null".to_string(),
            ));
        }
        Ok(Self { c_ptr })
    }

    pub fn set_current_stream(&self) -> Result<(), TchError> {
        unsafe_torch_err!(torch_sys::cuda::atcs_set_current_stream(self.c_ptr));
        Ok(())
    }
}

impl Drop for CudaStream {
    fn drop(&mut self) {
        unsafe_torch!(torch_sys::cuda::atcs_free(self.c_ptr))
    }
}
