use std::convert::From;

#[derive(Debug)]
pub struct TorchError {
    c_error: String,
}

impl TorchError {
    pub fn new(c_error: String) -> TorchError {
        TorchError { c_error }
    }
}

impl From<std::ffi::NulError> for TorchError {
    fn from(_err: std::ffi::NulError) -> Self {
        TorchError {
            c_error: "ffi nul error".to_string(),
        }
    }
}

extern "C" {
    fn get_and_reset_last_err() -> *mut libc::c_char;
}

pub(crate) fn read_and_clean_error() -> Result<(), TorchError> {
    unsafe {
        let torch_last_err = get_and_reset_last_err();
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

pub(crate) fn path_to_str(path: &std::path::Path) -> Result<&str, TorchError> {
    match path.to_str() {
        Some(path) => Ok(path),
        None => Err(TorchError::new(format!("path {:?} is none", path))),
    }
}
