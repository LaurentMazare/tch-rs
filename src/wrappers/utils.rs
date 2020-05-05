use anyhow::Result;
use libc::c_char;
use std::convert::From;

/// Errors returned by the torch C++ api.
#[derive(Debug)]
pub struct TorchError {
    c_error: String,
}

impl std::fmt::Display for TorchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "internal torch error {}", self.c_error)
    }
}

impl std::error::Error for TorchError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        None
    }
}
impl From<std::ffi::NulError> for TorchError {
    fn from(_err: std::ffi::NulError) -> Self {
        TorchError {
            c_error: "ffi nul error".to_string(),
        }
    }
}

// This returns None on the null pointer. If not null, the pointer gets
// freed.
pub(super) unsafe fn ptr_to_string(ptr: *mut c_char) -> Option<String> {
    if !ptr.is_null() {
        let str = std::ffi::CStr::from_ptr(ptr).to_string_lossy().into_owned();
        libc::free(ptr as *mut libc::c_void);
        Some(str)
    } else {
        None
    }
}

pub(super) fn read_and_clean_error() -> Result<(), TorchError> {
    unsafe {
        match ptr_to_string(torch_sys::get_and_reset_last_err()) {
            None => Ok(()),
            Some(c_error) => Err(TorchError { c_error }),
        }
    }
}

macro_rules! unsafe_torch {
    ($e:expr) => {{
        let v = unsafe { $e };
        crate::wrappers::utils::read_and_clean_error().unwrap();
        v
    }};
}

macro_rules! unsafe_torch_err {
    ($e:expr) => {{
        let v = unsafe { $e };
        crate::wrappers::utils::read_and_clean_error()?;
        v
    }};
}

// Be cautious when using this function as the returned CString should be stored
// in a variable when using as_ptr. Otherwise dangling pointer issues are likely
// to happen.
pub(super) fn path_to_cstring<T: AsRef<std::path::Path>>(path: T) -> Result<std::ffi::CString> {
    let path = path.as_ref();
    match path.to_str() {
        Some(path) => Ok(std::ffi::CString::new(path)?),
        None => Err(format_err!("path {:?} is none", path)),
    }
}

/// Sets the random seed used by torch.
pub fn manual_seed(seed: i64) {
    unsafe_torch!(torch_sys::at_manual_seed(seed))
}

/// Get the number of threads used by torch for inter-op parallelism.
pub fn get_num_interop_threads() -> i32 {
    unsafe_torch!(torch_sys::at_get_num_interop_threads())
}

/// Get the number of threads used by torch in parallel regions.
pub fn get_num_threads() -> i32 {
    unsafe_torch!(torch_sys::at_get_num_threads())
}

/// Set the number of threads used by torch for inter-op parallelism.
pub fn set_num_interop_threads(n_threads: i32) {
    unsafe_torch!(torch_sys::at_set_num_interop_threads(n_threads))
}

/// Set the number of threads used by torch in parallel regions.
pub fn set_num_threads(n_threads: i32) {
    unsafe_torch!(torch_sys::at_set_num_threads(n_threads))
}
