use crate::TchError;
use libc::c_char;
use std::io;

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

pub(super) fn read_and_clean_error() -> Result<(), TchError> {
    unsafe {
        match ptr_to_string(torch_sys::get_and_reset_last_err()) {
            None => Ok(()),
            Some(c_error) => Err(TchError::Torch(c_error)),
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
pub(super) fn path_to_cstring<T: AsRef<std::path::Path>>(
    path: T,
) -> Result<std::ffi::CString, TchError> {
    let path = path.as_ref();
    match path.to_str() {
        Some(path) => Ok(std::ffi::CString::new(path)?),
        None => Err(TchError::Io(io::Error::new(
            io::ErrorKind::Other,
            format!("path {:?} cannot be converted to UTF-8", path),
        ))),
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

/// Quantization engines
#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum QEngine {
    NoQEngine,
    FBGEMM,
    QNNPACK,
}

impl QEngine {
    fn to_cint(self) -> i32 {
        match self {
            QEngine::NoQEngine => 0,
            QEngine::FBGEMM => 1,
            QEngine::QNNPACK => 2,
        }
    }
    pub fn set(self) -> Result<(), TchError> {
        unsafe_torch_err!(torch_sys::at_set_qengine(self.to_cint()));
        Ok(())
    }
}
