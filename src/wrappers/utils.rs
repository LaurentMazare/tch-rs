use failure::Fallible;
use libc::c_char;
use std::convert::From;

/// Errors returned by the torch C++ api.
#[derive(Fail, Debug)]
#[fail(display = "Internal torch error: {}", c_error)]
pub struct TorchError {
    c_error: String,
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
pub(super) fn path_to_cstring<T: AsRef<std::path::Path>>(path: T) -> Fallible<std::ffi::CString> {
    let path = path.as_ref();
    match path.to_str() {
        Some(path) => Ok(std::ffi::CString::new(path)?),
        None => Err(format_err!("path {:?} is none", path)),
    }
}

/// Sets the random seed used by torch.
pub fn manual_seed(seed: i64) {
    unsafe_torch!({ torch_sys::at_manual_seed(seed) })
}
