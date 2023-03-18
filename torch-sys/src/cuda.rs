use libc::c_int;

#[repr(C)]
pub struct CCudaGraph {
    _private: [u8; 0],
}

#[repr(C)]
pub struct CCudaStream {
    _private: [u8; 0],
}

extern "C" {
    /// Returns the number of CUDA devices available.
    pub fn atc_cuda_device_count() -> c_int;

    /// Returns true if at least one CUDA device is available.
    pub fn atc_cuda_is_available() -> c_int;

    /// Returns true if CUDA is available, and CuDNN is available.
    pub fn atc_cudnn_is_available() -> c_int;

    /// Sets the seed for the current GPU.
    pub fn atc_manual_seed(seed: u64);

    /// Sets the seed for all available GPUs.
    pub fn atc_manual_seed_all(seed: u64);

    /// Waits for all kernels in all streams on a CUDA device to complete.
    pub fn atc_synchronize(device_index: i64);

    /// Returns true if CUDNN is enabled by the user.
    pub fn atc_user_enabled_cudnn() -> c_int;

    /// Enable or disable CUDNN.
    pub fn atc_set_user_enabled_cudnn(b: c_int);

    /// Sets CUDNN benchmark mode.
    pub fn atc_set_benchmark_cudnn(b: c_int);

    pub fn atcg_new() -> *mut CCudaGraph;
    pub fn atcg_free(arg: *mut CCudaGraph);
    pub fn atcg_replay(arg: *mut CCudaGraph);
    pub fn atcg_reset(arg: *mut CCudaGraph);
    pub fn atcg_capture_begin(arg: *mut CCudaGraph);
    pub fn atcg_capture_end(arg: *mut CCudaGraph);

    pub fn atcs_free(arg: *mut CCudaStream);
    pub fn atcs_get_stream_from_pool(high_priority: c_int, device: c_int) -> *mut CCudaStream;
    pub fn atcs_get_default_stream(device: c_int) -> *mut CCudaStream;
    pub fn atcs_get_current_stream(device: c_int) -> *mut CCudaStream;
    pub fn atcs_set_current_stream(arg: *mut CCudaStream);
}
