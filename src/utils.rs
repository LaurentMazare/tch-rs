use failure::Fallible;
use std::convert::From;

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

pub(crate) fn read_and_clean_error() -> Result<(), TorchError> {
    unsafe {
        let torch_last_err = torch_sys::get_and_reset_last_err();
        if !torch_last_err.is_null() {
            let c_error = std::ffi::CStr::from_ptr(torch_last_err)
                .to_string_lossy()
                .into_owned();
            libc::free(torch_last_err as *mut libc::c_void);
            Err(TorchError { c_error })
        } else {
            Ok(())
        }
    }
}

macro_rules! unsafe_torch {
    ($e:expr) => {{
        let v = unsafe { $e };
        crate::utils::read_and_clean_error().unwrap();
        v
    }};
}

macro_rules! unsafe_torch_err {
    ($e:expr) => {{
        let v = unsafe { $e };
        crate::utils::read_and_clean_error()?;
        v
    }};
}

// Be cautious when using this function as the returned CString should be stored
// in a variable when using as_ptr. Otherwise dangling pointer issues are likely
// to happen.
pub(crate) fn path_to_cstring<T: AsRef<std::path::Path>>(path: T) -> Fallible<std::ffi::CString> {
    let path = path.as_ref();
    match path.to_str() {
        Some(path) => Ok(std::ffi::CString::new(path)?),
        None => Err(format_err!("path {:?} is none", path)),
    }
}
