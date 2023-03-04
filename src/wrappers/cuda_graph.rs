//! CUDA Graph API.

use crate::TchError;

pub struct CudaGraph {
    c_ptr: *mut torch_sys::cuda::CCudaGraph,
}

impl CudaGraph {
    pub fn new() -> Result<Self, TchError> {
        let c_ptr = unsafe_torch_err!(torch_sys::cuda::atcg_new());
        if c_ptr.is_null() {
            return Err(TchError::Torch("CudaGraph::new() returned null".to_string()));
        }
        Ok(Self { c_ptr })
    }

    pub fn capture_begin(&mut self) -> Result<(), TchError> {
        unsafe_torch_err!(torch_sys::cuda::atcg_capture_begin(self.c_ptr));
        Ok(())
    }

    pub fn capture_end(&mut self) -> Result<(), TchError> {
        unsafe_torch_err!(torch_sys::cuda::atcg_capture_end(self.c_ptr));
        Ok(())
    }

    pub fn replay(&mut self) -> Result<(), TchError> {
        unsafe_torch_err!(torch_sys::cuda::atcg_replay(self.c_ptr));
        Ok(())
    }

    pub fn reset(&mut self) -> Result<(), TchError> {
        unsafe_torch_err!(torch_sys::cuda::atcg_reset(self.c_ptr));
        Ok(())
    }
}

impl Drop for CudaGraph {
    fn drop(&mut self) {
        unsafe_torch!(torch_sys::cuda::atcg_free(self.c_ptr))
    }
}
