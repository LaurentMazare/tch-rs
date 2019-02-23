use crate::utils::read_and_clean_error;
use libc::c_int;
extern "C" {
    fn atc_get_num_threads() -> c_int;
    fn atc_set_num_threads(n: c_int);
    fn atc_cuda_device_count() -> c_int;
    fn atc_cuda_is_available() -> c_int;
    fn atc_cudnn_is_available() -> c_int;
    fn atc_set_benchmark_cudnn(b: c_int);
}

pub fn get_num_threads() -> i64 {
    let res = unsafe_torch!({ atc_get_num_threads() });
    i64::from(res)
}

pub fn set_num_threads(num_threads: i64) {
    unsafe_torch!({ atc_set_num_threads(num_threads as i32) })
}

pub fn cuda_device_count() -> i64 {
    let res = unsafe_torch!({ atc_cuda_device_count() });
    i64::from(res)
}

pub fn cuda_is_available() -> bool {
    unsafe_torch!({ atc_cuda_is_available() }) != 0
}

pub fn cudnn_is_available() -> bool {
    unsafe_torch!({ atc_cudnn_is_available() }) != 0
}

pub fn set_benchmark_cudnn(b: bool) {
    unsafe_torch!({ atc_set_benchmark_cudnn(if b { 1 } else { 0 }) })
}
